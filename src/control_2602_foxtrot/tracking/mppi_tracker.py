#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import numpy as np
import math
import time

class MPPITracker(Node):
    def __init__(self):
        super().__init__('mppi_tracker') 
        
        # =======================================================
        self.declare_parameter('control_frequency', 50.0) 
        self.declare_parameter('wheelbase', 0.42)          
        self.declare_parameter('max_steering_angle', 2.0)  
        
        self.declare_parameter('max_velocity', 10.5)
        self.declare_parameter('min_velocity', 3.4)
        self.declare_parameter('max_acceleration', 8.0)
        self.declare_parameter('max_deceleration', 10.0)
        
        self.declare_parameter('k_stanley', 6.0)
        self.declare_parameter('k_soft', 1.0)# Reducido para que responda rápido a velocidades bajas
        
        self.declare_parameter('repulsion_radius', 1.0)   
        self.declare_parameter('repulsion_gain', 1.7)      

        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ekf/odometry', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, rclpy.qos.qos_profile_sensor_data)
        from std_msgs.msg import Int32, Empty
        self.laps_sub = self.create_subscription(Int32, '/laps_cmd', self.laps_callback, 10)
        
        # Publishers
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_raw', 10)
        self.lap_comp_pub = self.create_publisher(Empty, '/lap_completed', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.global_path = None
        self.current_vel = 0.0
        self.cmd_v_prev = 0.0
        
        self.laps_remaining = 0
        self.prev_closest_idx = 0 
        
        self.closest_obstacle_dist = 999.0
        self.closest_obstacle_angle = 0.0
        
        self.dt = 1.0 / self.get_parameter('control_frequency').value
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        self.get_logger().info("Controlador STANLEY en línea. Freno Ortogonal (Anti-Orbiting) activado.")

    def laps_callback(self, msg):
        self.laps_remaining = msg.data
        if self.laps_remaining > 0:
            self.get_logger().info(f"Modo Carrera F1 Activado: {self.laps_remaining} vueltas restantes.")

    def path_callback(self, msg):
        if len(msg.poses) < 2:
            return
        path = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
        
        # If the path is a loop (or almost closed), connect the ends to make it perfectly cyclic!
        dist_end_to_start = math.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
        if dist_end_to_start < 5.0 and dist_end_to_start > 0.1:
            num_interp = int(dist_end_to_start / 0.05)
            for i in range(1, num_interp):
                t = i / num_interp
                nx = path[-1][0] * (1 - t) + path[0][0] * t
                ny = path[-1][1] * (1 - t) + path[0][1] * t
                path.append([nx, ny])
                
        self.global_path = np.array(path)

    def odom_callback(self, msg):
        self.current_vel = msg.twist.twist.linear.x

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        min_range = 0.4 # Ignora el chasis del robot
        
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
        
        # Local search to prevent jumping across the loop
        if self.prev_closest_idx > 0 and len(self.global_path) > 100:
            search_indices = [(self.prev_closest_idx + i) % len(self.global_path) for i in range(-20, 50)]
            local_dists = dists[search_indices]
            min_local_idx = int(np.argmin(local_dists))
            closest_idx = search_indices[min_local_idx]
        else:
            closest_idx = int(np.argmin(dists))
        
        # Lap detection: if we jump from end of path to beginning of path
        if closest_idx < 50 and self.prev_closest_idx > len(self.global_path) - 50:
            if self.laps_remaining > 0:
                self.laps_remaining -= 1
                self.get_logger().info(f"Vuelta completada! Faltan {self.laps_remaining} vueltas.")
                from std_msgs.msg import Empty
                msg = Empty()
                self.lap_comp_pub.publish(msg)
                
        self.prev_closest_idx = closest_idx
            
        goal_x = self.global_path[-1, 0]
        goal_y = self.global_path[-1, 1]
        dist_to_goal = math.hypot(goal_x - fx, goal_y - fy)
        
        # =======================================================
        # FRENO ORTOGONAL & LAPS LOGIC
        # =======================================================
        dx_goal = goal_x - fx
        dy_goal = goal_y - fy
        dot_product = dx_goal * math.cos(ryaw) + dy_goal * math.sin(ryaw)
        passed_goal = (dot_product < 0.0) and (dist_to_goal < 1.0)
        
        # Only stop if we are on the FINAL lap!
        if self.laps_remaining <= 0:
            if dist_to_goal < 0.35 or passed_goal:
                self.stop_vehicle()
                if passed_goal:
                    self.get_logger().info("Meta superada. Anclaje de emergencia activado.")
                else:
                    self.get_logger().info("Carrera Finalizada. Estacionamiento preciso completado.")
                return
        else:
            # If we are doing laps, override the distance to goal so it NEVER brakes!
            dist_to_goal = 999.0
            
        # Circular next point for steering
        next_idx = (closest_idx + 1) % len(self.global_path)
        px = self.global_path[closest_idx, 0]
        py = self.global_path[closest_idx, 1]
        px_next = self.global_path[next_idx, 0]
        py_next = self.global_path[next_idx, 1]
        
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

        # Reactive Obstacle Avoidance (Potential Field)
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
        min_v = 2.5 # Reduced from 3.4 so it takes sharp corners safely without slipping
        creep_v = 0.6 
        
        lookahead_dist = max(6.0, max_v * 0.7) # Look ahead further at high speeds
        
        # 2. MODULACIÓN DE VELOCIDAD (Frenado en curvas circulares)
        if dist_to_goal < 4.0:
            target_vel = max(creep_v, min_v * (dist_to_goal / 4.0))
        else:
            future_idx = closest_idx
            dist_accum = 0.0
            for i in range(1, int(lookahead_dist / 0.05) + 50):
                idx1 = (closest_idx + i - 1) % len(self.global_path)
                idx2 = (closest_idx + i) % len(self.global_path)
                dist_accum += math.hypot(self.global_path[idx2, 0] - self.global_path[idx1, 0], 
                                         self.global_path[idx2, 1] - self.global_path[idx1, 1])
                if dist_accum > lookahead_dist:
                    future_idx = idx2
                    break
                    
            future_next = (future_idx + 1) % len(self.global_path)
            future_yaw = math.atan2(self.global_path[future_next, 1] - self.global_path[future_idx, 1],
                                    self.global_path[future_next, 0] - self.global_path[future_idx, 0])
                
            yaw_diff = abs(math.atan2(math.sin(future_yaw - ryaw), math.cos(future_yaw - ryaw)))
            # Brake harder and earlier for curves (0.35 rad diff ~20 deg will brake to min_v)
            speed_factor = max(0.0, 1.0 - (yaw_diff / 0.35))
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