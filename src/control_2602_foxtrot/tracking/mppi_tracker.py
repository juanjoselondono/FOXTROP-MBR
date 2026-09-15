#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import numpy as np
import math
import time

class MPPITracker(Node):
    def __init__(self):
        super().__init__('mppi_tracker') 
        
        # =======================================================
        # DINÁMICA STANLEY (Parámetros Mecánicos de Precisión)
        # =======================================================
        self.declare_parameter('control_frequency', 50.0) 
        self.declare_parameter('wheelbase', 0.42)          
        self.declare_parameter('max_steering_angle', 2.0)  
        
        self.declare_parameter('max_velocity', 8.0)       # FULL SEND F1 MODE
        self.declare_parameter('min_velocity', 3.0)        # Never crawl
        self.declare_parameter('max_acceleration', 5.0)   # Aggressive launch
        self.declare_parameter('max_deceleration', 8.0)   # Hard braking for corners
        
        self.declare_parameter('k_stanley', 4.0)           # Tuned for high-speed stability
        self.declare_parameter('k_soft', 1.0)              # Reducido para que responda rápido a velocidades bajas
        
        self.declare_parameter('repulsion_radius', 1.0)   
        self.declare_parameter('repulsion_gain', 1.7)      

        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ekf/odometry', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, rclpy.qos.qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_raw', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.global_path = None
        self.current_vel = 0.0
        self.cmd_v_prev = 0.0 
        
        self.closest_obstacle_dist = 999.0
        self.closest_obstacle_angle = 0.0
        
        self.dt = 1.0 / self.get_parameter('control_frequency').value
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info("Controlador STANLEY en línea. Freno Ortogonal (Anti-Orbiting) activado.")

    def path_callback(self, msg):
        if len(msg.poses) < 2:
            return
        self.global_path = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])

    def odom_callback(self, msg):
        self.current_vel = msg.twist.twist.linear.x

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        min_range = 0.15
        
        valid = (ranges >= min_range) & (ranges < 5.0) & np.isfinite(ranges)
        if np.any(valid):
            valid_ranges = ranges[valid]
            valid_angles = angles[valid]
            min_idx = np.argmin(valid_ranges)
            self.closest_obstacle_dist = valid_ranges[min_idx]
            self.closest_obstacle_angle = valid_angles[min_idx]
        else:
            self.closest_obstacle_dist = 999.0

    def get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def stop_vehicle(self):
        decel = self.get_parameter('max_deceleration').value
        self.cmd_v_prev = max(0.0, self.cmd_v_prev - decel * self.dt)
        
        if self.cmd_v_prev < 0.05:
            self.cmd_v_prev = 0.0
            
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(self.cmd_v_prev)
        msg.twist.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def control_loop(self):
        t0 = time.perf_counter()
        
        if self.global_path is None or len(self.global_path) < 2:
            self.stop_vehicle()
            return
            
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            rx = trans.transform.translation.x
            ry = trans.transform.translation.y
            ryaw = self.get_yaw(trans.transform.rotation)
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.stop_vehicle()
            return

        L = self.get_parameter('wheelbase').value
        max_steer = self.get_parameter('max_steering_angle').value
        
        fx = rx + L * math.cos(ryaw)
        fy = ry + L * math.sin(ryaw)

        dists = np.hypot(self.global_path[:, 0] - fx, self.global_path[:, 1] - fy)
        closest_idx = int(np.argmin(dists))
        
        if closest_idx >= len(self.global_path) - 1:
            closest_idx = len(self.global_path) - 2
            
        goal_x = self.global_path[-1, 0]
        goal_y = self.global_path[-1, 1]
        dist_to_goal = math.hypot(goal_x - fx, goal_y - fy)
        
        # =======================================================
        # FRENO ORTOGONAL (Evitar el bucle de "Goal Orbiting")
        # =======================================================
        # Calcular si la meta ha quedado detrás del morro del vehículo
        dx_goal = goal_x - fx
        dy_goal = goal_y - fy
        dot_product = dx_goal * math.cos(ryaw) + dy_goal * math.sin(ryaw)
        passed_goal = (dot_product < 0.0) and (dist_to_goal < 1.0)
        
        # Condición de ancla extendida (0.4m) o superación física de la meta
        if dist_to_goal < 0.4 or closest_idx >= len(self.global_path) - 2 or passed_goal:
            self.stop_vehicle()
            if passed_goal:
                self.get_logger().info("Meta superada. Anclaje de emergencia activado.", throttle_duration_sec=2.0)
            else:
                self.get_logger().info("Estacionamiento preciso completado.", throttle_duration_sec=2.0)
            return
            
        px = self.global_path[closest_idx, 0]
        py = self.global_path[closest_idx, 1]
        
        px_next = self.global_path[closest_idx + 1, 0]
        py_next = self.global_path[closest_idx + 1, 1]
        
        path_yaw = math.atan2(py_next - py, px_next - px)
        
        theta_e = path_yaw - ryaw
        theta_e = math.atan2(math.sin(theta_e), math.cos(theta_e)) 
        
        cross_prod = (px_next - px) * (fy - py) - (py_next - py) * (fx - px)
        e_y = math.copysign(dists[closest_idx], cross_prod)
        
        k_stanley = self.get_parameter('k_stanley').value
        k_soft = self.get_parameter('k_soft').value
        
        stanley_steer = theta_e + math.atan2(-k_stanley * e_y, abs(self.current_vel) + k_soft)
        
        # Atenuación de dirección terminal (evita zig-zag al llegar)
        if dist_to_goal < 1.5:
            stanley_steer *= (dist_to_goal / 1.5)

        repulsion_radius = self.get_parameter('repulsion_radius').value
        repulsion_gain = self.get_parameter('repulsion_gain').value
        
        if self.closest_obstacle_dist < repulsion_radius:
            repulsion = (repulsion_radius - self.closest_obstacle_dist) * repulsion_gain
            if self.closest_obstacle_angle > 0: 
                stanley_steer -= repulsion      
            else:                               
                stanley_steer += repulsion      

        steer_angle = float(np.clip(stanley_steer, -max_steer, max_steer))

        # =======================================================
        # PERFILADO DE VELOCIDAD DE ARRASTRE (Creep Velocity)
        # =======================================================
        max_v = self.get_parameter('max_velocity').value
        min_v = self.get_parameter('min_velocity').value
        creep_v = 0.6 
        # 2. MODULACIÓN DE VELOCIDAD (Frenado en curvas)
        if dist_to_goal < 4.0:
            target_vel = max(creep_v, min_v * (dist_to_goal / 4.0))
        else:
            future_idx = closest_idx
            for i in range(closest_idx, len(self.global_path)):
                if math.hypot(self.global_path[i, 0] - fx, self.global_path[i, 1] - fy) > 6.0:
                    future_idx = i
                    break
                    
            if future_idx < len(self.global_path) - 1:
                future_yaw = math.atan2(self.global_path[future_idx+1, 1] - self.global_path[future_idx, 1],
                                        self.global_path[future_idx+1, 0] - self.global_path[future_idx, 0])
            else:
                future_yaw = path_yaw
                
            yaw_diff = abs(math.atan2(math.sin(future_yaw - ryaw), math.cos(future_yaw - ryaw)))
            speed_factor = max(0.0, 1.0 - (yaw_diff / 0.6))
            target_vel = min_v + speed_factor * (max_v - min_v)

        accel = self.get_parameter('max_acceleration').value
        decel = self.get_parameter('max_deceleration').value
        
        if target_vel > self.cmd_v_prev:
            self.cmd_v_prev = min(target_vel, self.cmd_v_prev + accel * self.dt)
        else:
            self.cmd_v_prev = max(target_vel, self.cmd_v_prev - decel * self.dt)
            
        cmd_v = self.cmd_v_prev

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(cmd_v)
        msg.twist.angular.z = steer_angle
        self.cmd_pub.publish(msg)

        comp_time = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(
            f"STANLEY | D2G:{dist_to_goal:.2f}m | tgt:{target_vel:.2f} vel:{cmd_v:.2f} str:{steer_angle:.2f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = MPPITracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()