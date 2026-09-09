#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import TwistStamped  # <-- Cambio a mensaje estampado
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
import math

class PIDTracker(Node):
    def __init__(self):
        super().__init__('pid_pt_2602_foxtrot', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.path_sub = self.create_subscription(Path, '/planned_path', self.path_cb, 10)
        
        # Publicador compatible con el estándar ROS 2 Jazzy ros2_control
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
        self.get_logger().info("Controlador PID activo. Firmas de tiempo habilitadas.")

    def path_cb(self, msg):
        self.path = msg
        self.get_logger().info(f"Ruta cargada exitosamente: {len(msg.poses)} waypoints.")

    def control_loop(self):
        if not self.path or not self.path.poses:
            return
            
        try:
            source_frame = self.path.header.frame_id
            if not source_frame:
                source_frame = 'map'

            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', source_frame, rclpy.time.Time())
            except Exception as tf_error:
                self.get_logger().warn(f"Bloqueo TF ({source_frame} -> base_link): {tf_error}")
                return

            last_pose = self.path.poses[-1].pose
            last_pose_transformed = tf2_geometry_msgs.do_transform_pose(last_pose, transform)
            
            dist_to_goal = math.hypot(last_pose_transformed.position.x, last_pose_transformed.position.y)
            
            if dist_to_goal < 0.3:
                self.get_logger().info(">>> META ALCANZADA <<<")
                self.path = None
                self.publish_stop()
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
                    
            if not target_point:
                target_point = last_pose_transformed

            error = target_point.position.y 
            
            kp = self.get_parameter('kp').value
            ki = self.get_parameter('ki').value
            kd = self.get_parameter('kd').value
            
            self.error_integral += error * 0.05
            self.error_integral = max(min(self.error_integral, 1.0), -1.0)
            derivative = (error - self.prev_error) / 0.05
            self.prev_error = error
            
            steering = (kp * error) + (ki * self.error_integral) + (kd * derivative)
            angular_z = max(min(steering, 1.5), -1.5)
            
            speed_penalty = 1.0 - (0.5 * (abs(angular_z) / 1.5))
            linear_x = self.get_parameter('v_nominal').value * speed_penalty
            
            # Construcción de mensaje estampado (TwistStamped)
            cmd = TwistStamped()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.header.frame_id = 'base_link'
            cmd.twist.linear.x = float(linear_x)
            cmd.twist.angular.z = float(angular_z)
            self.cmd_pub.publish(cmd)
            
            self.get_logger().info(f"Motores [ v: {linear_x:.2f} | w: {angular_z:.2f} ] Dist. Restante: {dist_to_goal:.2f}m", throttle_duration_sec=0.5)

        except Exception as e:
            self.get_logger().error(f"Falla matemática interna: {e}")

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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()