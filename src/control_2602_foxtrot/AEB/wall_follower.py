#!/usr/bin/env python3
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import math

class WallFollowerNode(Node):
    def __init__(self):
        super().__init__('wall_follower_node')
        
        self.traj_d = 0.5       
        self.lookahead_L = 0.5    
        self.theta_deg = 45.0     
        
        self.kp = 1.5             
        self.kd = 0.1             
        self.forward_velocity = 0.8  # Increased baseline speed
        
        self.prev_error = 0.0
        self.last_time = 0.0      # Initialized to zero for clock sync
        self.wf_enabled = False
        
        self.state_sub = self.create_subscription(
            Bool, '/wf/state', self.state_callback, 10)
            
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
            
        # FIXED: twist_mux expects standard Twist on its input channels
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_nav', 10)            
        self.get_logger().info("PD Wall Follower online. Synchronized to Simulation Clock.")

    def state_callback(self, msg):
        if self.wf_enabled and not msg.data:
            self.stop_vehicle()
        self.wf_enabled = msg.data

    def get_range_at_angle(self, msg, angle_deg):
        angle_rad = math.radians(angle_deg)
        if angle_rad < msg.angle_min or angle_rad > msg.angle_max:
            return float('inf')
            
        index = int((angle_rad - msg.angle_min) / msg.angle_increment)
        r = msg.ranges[index]
        if math.isinf(r) or math.isnan(r) or r < msg.range_min or r > msg.range_max:
            return float('inf')
        return r

    def scan_callback(self, msg):
        if not self.wf_enabled:
            self.last_time = 0.0 # Reset clock logic when disabled
            return

        # FIXED: Extracting time from the ROS 2 clock instead of the Python system clock
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Skip the first iteration to establish a valid time delta
        if self.last_time == 0.0:
            self.last_time = current_time
            return
            
        dt = current_time - self.last_time
        if dt <= 0.0:
            return

        b = self.get_range_at_angle(msg, -90.0)
        a = self.get_range_at_angle(msg, -90.0 + self.theta_deg)

        if math.isinf(a) or math.isinf(b):
            self.get_logger().warn("Wall Follower: Target surface lost on RIGHT side. Halting.", throttle_duration_sec=2.0)
            self.stop_vehicle()
            return

        theta_rad = math.radians(self.theta_deg)
        numerator = (a * math.cos(theta_rad)) - b
        denominator = a * math.sin(theta_rad)
        
        if abs(denominator) < 1e-6:
            alpha = 0.0
        else:
            alpha = math.atan(numerator / denominator)

        ab_distance = b * math.cos(alpha)
        future_distance = ab_distance + (self.lookahead_L * math.sin(alpha))
        error = self.traj_d - future_distance
        
        # Derivative is now mathematically stable
        error_dot = (error - self.prev_error) / dt
        theta_d = (self.kp * error) + (self.kd * error_dot)
        
        self.prev_error = error
        self.last_time = current_time
        
        self.publish_control(theta_d)
    def publish_control(self, steering_effort):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = self.forward_velocity
        cmd.twist.angular.z = float(steering_effort)
        self.cmd_pub.publish(cmd)
    
    def stop_vehicle(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()