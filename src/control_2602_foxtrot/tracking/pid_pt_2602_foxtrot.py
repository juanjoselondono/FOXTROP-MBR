#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
import math

class PIDTracker(Node):
    def __init__(self):
        super().__init__('pid_pt_2602_foxtrot')
        self.path_sub = self.create_subscription(Path, '/planned_path', self.path_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/diffdrive_controller/cmd_vel', 10)
        
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
        self.get_logger().info("Nueva ruta RRT recibida. Iniciando seguimiento.")

    def control_loop(self):
        if not self.path or not self.path.poses:
            self.publish_stop()
            return
            
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', self.path.header.frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.5))
        except Exception:
            return

        # Verificar si ya llegamos al destino final (Tolerancia de 0.3 metros)
        last_pose_transformed = tf2_geometry_msgs.do_transform_pose(self.path.poses[-1], transform)
        dist_to_goal = math.hypot(last_pose_transformed.pose.position.x, last_pose_transformed.pose.position.y)
        
        if dist_to_goal < 0.3:
            self.get_logger().info("¡Meta alcanzada! Deteniendo vehículo.", throttle_duration_sec=2.0)
            self.path = None
            self.publish_stop()
            return

        lookahead = self.get_parameter('lookahead').value
        target_point = None
        
        for pose_stamped in self.path.poses:
            t_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
            x = t_pose.pose.position.x
            y = t_pose.pose.position.y
            distance = math.hypot(x, y)
            
            if x > 0 and distance >= lookahead:
                target_point = t_pose.pose
                break
                
        if not target_point:
            target_point = last_pose_transformed.pose

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
        
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)

    def publish_stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PIDTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()