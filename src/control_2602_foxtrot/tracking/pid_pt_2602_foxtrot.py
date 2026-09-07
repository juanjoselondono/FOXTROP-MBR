#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import TwistStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
import math

class PIDTracker(Node):
    def __init__(self):
        super().__init__('pid_pt_2602_foxtrot')
        self.path_sub = self.create_subscription(Path, '/planned_path', self.path_cb, 10)
        
        # Inyección Directa: Bypass del twist_mux con el formato estricto de ROS 2
        self.cmd_pub = self.create_publisher(TwistStamped, '/diffdrive_controller/cmd_vel', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.declare_parameter('kp', 0.8)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.2)
        self.declare_parameter('v_nominal', 0.8)
        self.declare_parameter('lookahead', 0.6)
        
        self.path = None
        self.error_integral = 0.0
        self.prev_error = 0.0
        
        self.timer = self.create_timer(0.05, self.control_loop)

    def path_cb(self, msg):
        self.path = msg

    def control_loop(self):
        if not self.path:
            return
            
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', self.path.header.frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.5))
        except Exception:
            return

        lookahead = self.get_parameter('lookahead').value
        target_point = None
        
        for pose_stamped in self.path.poses:
            t_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped.pose, transform)
            x = t_pose.position.x
            y = t_pose.position.y
            distance = math.hypot(x, y)
            
            if x > 0 and distance >= lookahead:
                target_point = t_pose
                break
                
        if not target_point and len(self.path.poses) > 0:
            target_point = tf2_geometry_msgs.do_transform_pose(self.path.poses[-1].pose, transform)

        if not target_point:
            self.publish_stop()
            return

        error = target_point.position.y 
        
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        
        self.error_integral += error * 0.05
        self.error_integral = max(min(self.error_integral, 1.0), -1.0)
        derivative = (error - self.prev_error) / 0.05
        self.prev_error = error
        
        steering_angular_velocity = (kp * error) + (ki * self.error_integral) + (kd * derivative)
        angular_z = max(min(steering_angular_velocity, 1.5), -1.5)
        
        speed_penalty = 1.0 - (0.5 * (abs(angular_z) / 1.5))
        linear_x = self.get_parameter('v_nominal').value * speed_penalty
        
        # Reensamblaje del TwistStamped
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = float(linear_x)
        cmd.twist.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PIDTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()