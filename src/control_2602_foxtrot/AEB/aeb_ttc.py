#!/usr/bin/env python3
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
import math

class KinematicAEBFilter(Node):
    def __init__(self):
        super().__init__('kinematic_aeb_filter')
        
        # User-Tuned Kinematic Parameters
        self.max_deceleration = 0.5  
        self.system_latency = 0.3    
        
        # Physical footprint parameters
        self.safety_margin = 0.3     
        self.corridor_width = 0.3    
        self.ttc_threshold = 1.5     
        self.strict_lockout_dist = 0.35 
        
        # ESC State Machine & History
        self.brake_state = 'IDLE'    
        self.brake_start_time = None
        self.active_brake_cmd = 0.0  
        self.dynamic_pulse_duration = 0.5 
        self.last_cmd_vel_forward = 0.0  # Tracks previous velocity to prevent standstill-reversing
        
        self.latest_scan = None
        self.aeb_enabled = True
        self.filtered_accel_x = 0.0  
        
        # QoS Profile: depth=1 prevents runaway buffer bloat after Wi-Fi lag spikes
        teleop_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscriptions
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.cmd_raw_sub = self.create_subscription(TwistStamped, '/cmd_vel_raw', self.cmd_raw_callback, teleop_qos)
        self.state_sub = self.create_subscription(Bool, '/aeb/state', self.state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data_raw', self.imu_callback, qos_profile_sensor_data)
            
        # Publishers 
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_stamped', 10)
        self.req_decel_pub = self.create_publisher(Float32, '/required_deceleration', 10)
        self.ttc_pub = self.create_publisher(Float32, '/ttc', 10) 
        
        self.get_logger().info("Elite AEB Online: Anti-Runaway QoS, Inertia Fix, and Standstill Tracker active.")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def state_callback(self, msg):
        self.aeb_enabled = msg.data

    def imu_callback(self, msg):
        alpha = 0.1
        self.filtered_accel_x = (alpha * msg.linear_acceleration.x) + ((1.0 - alpha) * self.filtered_accel_x)

    def calculate_forward_distance(self):
        if not self.latest_scan:
            return float('inf')
            
        min_dist = float('inf')
        for i, r in enumerate(self.latest_scan.ranges):
            if math.isinf(r) or math.isnan(r) or r < 0.15:
                continue
                
            angle = self.latest_scan.angle_min + (i * self.latest_scan.angle_increment)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            
            if x > 0.0 and abs(y) <= self.corridor_width / 2.0:
                if r < min_dist:
                    min_dist = r
        return min_dist

    def cmd_raw_callback(self, msg):
        current_time = self.get_clock().now()
        
        filtered_msg = TwistStamped()
        filtered_msg.header = msg.header
        filtered_msg.header.stamp = current_time.to_msg()
        filtered_msg.twist.angular = msg.twist.angular
        
        # 1. ANTI-RUNAWAY WATCHDOG (Stale Message Check)
        msg_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        now_time = current_time.nanoseconds * 1e-9
        
        # If message is older than 0.3s due to Wi-Fi lag, kill throttle to prevent a crash
        if abs(now_time - msg_time) > 0.3:
            self.get_logger().warn("NETWORK LAG DETECTED: Dropping stale command to prevent runaway.")
            cmd_vel_forward = 0.0
        else:
            cmd_vel_forward = msg.twist.linear.x

        front_distance = self.calculate_forward_distance()

        # 2. NOMINAL BYPASS 
        if not self.aeb_enabled or cmd_vel_forward <= 0.0:
            self.brake_state = 'IDLE'
            filtered_msg.twist.linear.x = cmd_vel_forward
            self.cmd_pub.publish(filtered_msg)
            self.last_cmd_vel_forward = cmd_vel_forward
            return

        # 3. KINEMATICS & IMU MATH
        ttc_value = front_distance / cmd_vel_forward if front_distance != float('inf') else float('inf')
        self.ttc_pub.publish(Float32(data=float(ttc_value)))

        accel_buffer = max(0.0, self.filtered_accel_x * 0.15) 
        dynamic_margin = self.safety_margin + accel_buffer

        latency_distance = cmd_vel_forward * self.system_latency
        available_distance = max(0.0, front_distance - dynamic_margin - latency_distance)
        
        req_decel = (cmd_vel_forward ** 2) / (2.0 * available_distance) if available_distance > 0 else float('inf')
        self.req_decel_pub.publish(Float32(data=float(req_decel)))

        # 4. TRIGGER EVALUATION (Including Strict Lockout)
        trigger_lockout = (front_distance <= self.strict_lockout_dist)
        trigger_aeb = (req_decel > self.max_deceleration) or (ttc_value < self.ttc_threshold) or trigger_lockout

        # 5. STATE MACHINE
        if trigger_aeb:
            if self.brake_state == 'IDLE':
                # Check if we were already stopped. If so, DO NOT brake (prevents double-tap reverse)
                if self.last_cmd_vel_forward <= 0.0:
                    self.brake_state = 'HOLD'
                    self.get_logger().warn(f"AEB BLOCKED: Obstacle at {front_distance:.2f}m. Forward gear severed.")
                    filtered_msg.twist.linear.x = 0.0
                else:
                    self.brake_state = 'BRAKING'
                    self.brake_start_time = current_time
                    
                    # If we breached the strict lockout unexpectedly (inertia), force max brake
                    if trigger_lockout:
                        self.active_brake_cmd = -0.8
                        self.dynamic_pulse_duration = 0.6
                        self.get_logger().warn(f"EMERGENCY LOCKOUT BRAKE! Dist: {front_distance:.2f}m.")
                    else:
                        # Standard proportional braking
                        accel_penalty = max(0.0, self.filtered_accel_x * 0.1)
                        brake_intensity = 0.6 if req_decel == float('inf') else (req_decel * 0.25) + accel_penalty
                        self.active_brake_cmd = -float(max(0.2, min(0.8, brake_intensity)))
                        self.dynamic_pulse_duration = float(max(0.5, min(1.5, cmd_vel_forward * 1.5)))
                        self.get_logger().warn(f"AEB ENGAGED! Force: {self.active_brake_cmd:.2f}")
                        
                    filtered_msg.twist.linear.x = self.active_brake_cmd 
                
            elif self.brake_state == 'BRAKING':
                elapsed = (current_time - self.brake_start_time).nanoseconds / 1e9
                if elapsed < self.dynamic_pulse_duration:
                    filtered_msg.twist.linear.x = self.active_brake_cmd
                else:
                    self.brake_state = 'HOLD'
                    filtered_msg.twist.linear.x = 0.0
                    
            elif self.brake_state == 'HOLD':
                filtered_msg.twist.linear.x = 0.0 
        else:
            self.brake_state = 'IDLE'
            filtered_msg.twist.linear.x = cmd_vel_forward
            
        self.cmd_pub.publish(filtered_msg)
        self.last_cmd_vel_forward = cmd_vel_forward

def main(args=None):
    rclpy.init(args=args)
    node = KinematicAEBFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Terminating AEB Filter.")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()