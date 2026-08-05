#!/usr/bin/env python3
from std_msgs.msg import Bool
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32
from rclpy.qos import qos_profile_sensor_data
import math

class KinematicAEBFilter(Node):
    def __init__(self):
        super().__init__('kinematic_aeb_filter')
        
        self.max_deceleration = 2.0  
        self.safety_margin = 0.5     
        self.front_cone_angle = 1.5  # What it represents: The angular width of the monitoring zone directly ahead of the vehicle.
        
        self.latest_scan = None
        
        # SUBSCRIBER UPGRADED: Now natively expects TwistStamped from twist_mux
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.cmd_raw_sub = self.create_subscription(
            TwistStamped, '/cmd_vel_raw', self.cmd_raw_callback, 10)
            
        # Publishers
        self.cmd_pub = self.create_publisher(TwistStamped, '/diffdrive_controller/cmd_vel', 10)
        self.req_decel_pub = self.create_publisher(Float32, '/required_deceleration', 10)
        self.ttc_pub = self.create_publisher(Float32, '/ttc', 10) # TTC Telemetry Output
        
        self.get_logger().info("Kinematic AEB Filter online. Full TwistStamped pipeline and TTC telemetry active.")
        #Enabling logic
        self.aeb_enabled = True
        self.state_sub = self.create_subscription(
            Bool, '/aeb/state', self.state_callback, 10)

    def scan_callback(self, msg):
        self.latest_scan = msg

    def calculate_forward_distance(self):
        if not self.latest_scan:
            return float('inf')
            
        min_dist = float('inf')
        msg = self.latest_scan
        
        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or math.isnan(r) or r < 0.4: 
                continue
                
            angle = msg.angle_min + (i * msg.angle_increment)
            if abs(angle) <= self.front_cone_angle / 2.0:
                if r < min_dist:
                    min_dist = r
                    
        return min_dist
    def state_callback(self, msg):
        self.aeb_enabled = msg.data

    def cmd_raw_callback(self, msg):
        # BYPASS MODE: If AEB is disabled via joystick, pass raw commands directly
        if not self.aeb_enabled:
            filtered_msg = TwistStamped()
            filtered_msg.header = msg.header
            filtered_msg.header.stamp = self.get_clock().now().to_msg()
            filtered_msg.twist = msg.twist
            self.cmd_pub.publish(filtered_msg)
            return
        front_distance = self.calculate_forward_distance()
        
        # Extracting velocity from the nested TwistStamped structure
        cmd_vel_forward = msg.twist.linear.x
        
        # --- TTC Calculation & Publication ---
        if cmd_vel_forward > 0.0 and front_distance != float('inf'):
            ttc_value = front_distance / cmd_vel_forward
        else:
            ttc_value = float('inf')  # No collision risk if stopped, reversing, or no obstacle
            
        ttc_msg = Float32()
        ttc_msg.data = float(ttc_value)
        self.ttc_pub.publish(ttc_msg)
        # -------------------------------------
        
        filtered_msg = TwistStamped()
        filtered_msg.header = msg.header
        filtered_msg.header.stamp = self.get_clock().now().to_msg()
        
        filtered_msg.twist.angular = msg.twist.angular
        filtered_msg.twist.linear.y = msg.twist.linear.y
        filtered_msg.twist.linear.z = msg.twist.linear.z
        
        if cmd_vel_forward <= 0.0:
            filtered_msg.twist.linear.x = cmd_vel_forward
            self.cmd_pub.publish(filtered_msg)
            return
            
        available_distance = max(0.0, front_distance - self.safety_margin)
        
        if available_distance > 0:
            req_decel = (cmd_vel_forward ** 2) / (2.0 * available_distance)
        else:
            req_decel = float('inf')
            
        decel_msg = Float32()
        decel_msg.data = float(req_decel)
        self.req_decel_pub.publish(decel_msg)
        
        if req_decel > self.max_deceleration:
            safe_velocity = math.sqrt(2.0 * self.max_deceleration * available_distance)
            self.get_logger().warn(f"AEB INTERVENTION: Clamping velocity to {safe_velocity:.2f} m/s.")
            filtered_msg.twist.linear.x = safe_velocity
        else:
            filtered_msg.twist.linear.x = cmd_vel_forward
            
        self.cmd_pub.publish(filtered_msg)

def main(args=None):
    rclpy.init(args=args)
    node = KinematicAEBFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()