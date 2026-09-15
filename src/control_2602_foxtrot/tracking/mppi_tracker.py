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
from scipy.spatial import cKDTree
from rclpy.qos import qos_profile_sensor_data

class MPPITracker(Node):
    def __init__(self):
        super().__init__('mppi_tracker')
        
        # Parameters
        self.declare_parameter('num_samples', 300)
        self.declare_parameter('horizon_seconds', 1.5)
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('temperature', 0.5)
        
        self.declare_parameter('wheelbase', 0.33)
        self.declare_parameter('max_steering_angle', 0.6)
        self.declare_parameter('max_velocity', 4.0)
        self.declare_parameter('max_acceleration', 3.0)
        self.declare_parameter('max_deceleration', 4.0)
        
        # Costs — Path Tracking Strict profile
        self.declare_parameter('weight_path', 40.0)        # DOMINANT: stay on the path
        self.declare_parameter('weight_heading', 10.0)      # HIGH: face the right direction
        self.declare_parameter('weight_velocity', 5.0)      # MODERATE: reach v_target but don't sacrifice path
        self.declare_parameter('weight_steering', 2.0)      # Penalize excessive steering
        self.declare_parameter('weight_smoothness', 1.0)    # Penalize abrupt steering changes
        self.declare_parameter('weight_collision', 500.0)   # SOFT: delegate hard braking to AEB

        # Safety — soft avoidance radius (AEB handles the hard stop)
        self.declare_parameter('collision_radius', 0.30)
        self.declare_parameter('lidar_min_range', 0.4)     # Ignore self-collision below this

        # Subs / Pubs — Pipeline: MPPI → /cmd_vel_raw (TwistStamped) → AEB → controller
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ekf/odometry', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_raw', 10)
        
        # TF2 — resolve map→base_link for correct spatial alignment
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # State — cached for async operation
        self.global_path = None
        self.cached_local_ref = None   # Last valid local reference for async operation
        self.current_vel = 0.0
        self.kdtree = None
        
        # Compute dynamic params
        self.dt = 1.0 / self.get_parameter('control_frequency').value
        self.horizon_steps = int(self.get_parameter('horizon_seconds').value / self.dt)
        
        # Nominal control sequence (accel, steer)
        # SEED with small positive acceleration to break static inertia
        self.U_a = np.full(self.horizon_steps, 0.5)
        self.U_s = np.zeros(self.horizon_steps)
        
        self.a_std = 1.5   # Wide exploration for acceleration
        self.s_std = 0.4    # Moderate steering exploration
        
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(
            f"MPPI Tracker initialized | H={self.horizon_steps} steps | dt={self.dt:.3f}s | "
            f"Publishing TwistStamped → /cmd_vel_raw"
        )

    def path_callback(self, msg):
        self.global_path = msg.poses

    def odom_callback(self, msg):
        # Only read velocity from odometry. Pose comes from TF2 (map frame).
        self.current_vel = msg.twist.twist.linear.x

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        min_range = self.get_parameter('lidar_min_range').value
        valid = (ranges >= min_range) & (ranges < 10.0) & np.isfinite(ranges)
        ox = ranges[valid] * np.cos(angles[valid])
        oy = ranges[valid] * np.sin(angles[valid])
        
        if len(ox) > 0:
            self.kdtree = cKDTree(np.column_stack((ox, oy)))
        else:
            self.kdtree = None

    def get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def extract_local_reference(self, rx, ry, ryaw):
        if not self.global_path:
            return self.cached_local_ref  # Async: use last valid reference
        
        path_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in self.global_path])
        if len(path_xy) < 2:
            return self.cached_local_ref
            
        dists = np.linalg.norm(path_xy - np.array([rx, ry]), axis=1)
        closest_idx = np.argmin(dists)
        
        # Lookahead: advance index so the car anticipates curves
        # Protect: never leave fewer than 2 points in the remaining segment
        closest_idx = min(closest_idx + 3, max(0, len(path_xy) - 2))
        
        dist_to_goal = math.hypot(path_xy[-1, 0] - rx, path_xy[-1, 1] - ry)
        
        end_idx = min(closest_idx + self.horizon_steps + 10, len(path_xy))
        ref_segment = path_xy[closest_idx:end_idx]
        
        if len(ref_segment) < 2:
            return self.cached_local_ref
            
        dx = ref_segment[:, 0] - rx
        dy = ref_segment[:, 1] - ry
        
        c = math.cos(-ryaw)
        s = math.sin(-ryaw)
        local_x = dx * c - dy * s
        local_y = dx * s + dy * c
        local_path = np.column_stack((local_x, local_y))
        
        # Curvature-aware velocity profile
        dx_path = np.gradient(local_x)
        dy_path = np.gradient(local_y)
        ddx_path = np.gradient(dx_path)
        ddy_path = np.gradient(dy_path)
        
        denom = (dx_path**2 + dy_path**2)**1.5
        curvature = np.zeros_like(denom)
        mask = denom > 1e-6
        curvature[mask] = np.abs(dx_path[mask] * ddy_path[mask] - dy_path[mask] * ddx_path[mask]) / denom[mask]
        
        a_lat_max = 2.5
        v_max = self.get_parameter('max_velocity').value
        v_target = np.sqrt(a_lat_max / (curvature + 0.01))
        v_target = np.clip(v_target, 0.8, v_max)
        
        # Pad or trim to match horizon_steps
        H = self.horizon_steps
        if len(local_path) > H:
            local_path = local_path[:H]
            v_target = v_target[:H]
        else:
            pad_len = H - len(local_path)
            local_path = np.vstack((local_path, np.tile(local_path[-1], (pad_len, 1))))
            v_target = np.append(v_target, np.full(pad_len, 0.0))
        
        ref_yaw = np.arctan2(np.gradient(local_path[:, 1]), np.gradient(local_path[:, 0]))
            
        result = (local_path, v_target, ref_yaw, dist_to_goal)
        self.cached_local_ref = result  # Cache for async operation
        return result

    def stop_vehicle(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def control_loop(self):
        t0 = time.perf_counter()
        
        # Get robot pose in MAP frame via TF2
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            rx = trans.transform.translation.x
            ry = trans.transform.translation.y
            q = trans.transform.rotation
            ryaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        except (LookupException, ConnectivityException, ExtrapolationException):
            self.get_logger().warn("Esperando TF map->base_link...", throttle_duration_sec=2.0)
            self.stop_vehicle()
            return
        
        # Async: try to extract reference, fall back to cached
        ref = self.extract_local_reference(rx, ry, ryaw)
        if not ref:
            self.get_logger().warn("Falta ruta o está fuera de alcance. Deteniendo...", throttle_duration_sec=2.0)
            self.stop_vehicle()
            return
            
        local_path, v_ref, yaw_ref, dist_to_goal = ref
        
        if dist_to_goal < 0.3:
            self.stop_vehicle()
            self.get_logger().info("Goal reached! Stopping.")
            return

        N = self.get_parameter('num_samples').value
        H = self.horizon_steps
        dt = self.dt
        L = self.get_parameter('wheelbase').value
        v_max = self.get_parameter('max_velocity').value
        a_max = self.get_parameter('max_acceleration').value
        d_max = self.get_parameter('max_deceleration').value
        s_max = self.get_parameter('max_steering_angle').value
        
        # Generate perturbations
        noise_a = np.random.normal(0, self.a_std, (N, H))
        noise_s = np.random.normal(0, self.s_std, (N, H))
        
        A = np.clip(self.U_a + noise_a, -d_max, a_max)
        S = np.clip(self.U_s + noise_s, -s_max, s_max)
        
        # === Vectorized Kinematic Bicycle Rollout ===
        x = np.zeros((N, H))
        y = np.zeros((N, H))
        yaw = np.zeros((N, H))
        v = np.zeros((N, H))
        
        curr_v = np.full(N, max(self.current_vel, 0.0))
        curr_x = np.zeros(N)
        curr_y = np.zeros(N)
        curr_yaw = np.zeros(N)
        
        for t in range(H):
            curr_v = np.clip(curr_v + A[:, t] * dt, 0.0, v_max)
            curr_x = curr_x + curr_v * np.cos(curr_yaw) * dt
            curr_y = curr_y + curr_v * np.sin(curr_yaw) * dt
            curr_yaw = curr_yaw + (curr_v / L) * np.tan(S[:, t]) * dt
            
            x[:, t] = curr_x
            y[:, t] = curr_y
            yaw[:, t] = curr_yaw
            v[:, t] = curr_v
            
        # === Cost Function ===
        costs = np.zeros(N)
        
        # 1. Path tracking (squared distance to reference)
        w_path = self.get_parameter('weight_path').value
        dist_sq = (x - local_path[:, 0])**2 + (y - local_path[:, 1])**2
        costs += np.sum(dist_sq, axis=1) * w_path
        
        # 2. Velocity tracking (CRITICAL for breaking static inertia)
        w_vel = self.get_parameter('weight_velocity').value
        v_err = (v - v_ref)**2
        costs += np.sum(v_err, axis=1) * w_vel
        
        # 3. Heading tracking
        w_head = self.get_parameter('weight_heading').value
        yaw_err = np.square(np.arctan2(np.sin(yaw - yaw_ref), np.cos(yaw - yaw_ref)))
        costs += np.sum(yaw_err, axis=1) * w_head

        # 4. Soft obstacle avoidance (AEB handles hard braking)
        num_collisions = 0
        if self.kdtree is not None:
            w_coll = self.get_parameter('weight_collision').value
            r_coll = self.get_parameter('collision_radius').value
            
            pts_flat = np.column_stack((x.ravel(), y.ravel()))
            obs_dists, _ = self.kdtree.query(pts_flat, k=1)
            obs_dists = obs_dists.reshape(N, H)
            
            # SOFT gradient cost: closer = more expensive (not binary kill)
            # 1/d cost provides a smooth gradient that pushes trajectories away
            min_obs_per_traj = np.min(obs_dists, axis=1)
            costs += w_coll / (min_obs_per_traj + 0.1)
            
            # Only add a hard penalty if VERY close (half collision radius)
            hard_coll = min_obs_per_traj < (r_coll * 0.5)
            costs[hard_coll] += w_coll * 10.0
            num_collisions = int(np.sum(hard_coll))

        # 5. Control effort (LOW to allow aggressive driving)
        w_steer = self.get_parameter('weight_steering').value
        w_smooth = self.get_parameter('weight_smoothness').value
        costs += np.sum(S**2, axis=1) * w_steer
        costs += np.sum(np.diff(S, axis=1)**2, axis=1) * w_smooth
        
        # === MPPI Weighted Update ===
        lam = self.get_parameter('temperature').value
        beta = np.min(costs)
        weights = np.exp(-(1.0 / lam) * (costs - beta))
        eta = np.sum(weights) + 1e-10
        weights /= eta
        
        self.U_a += np.sum(weights[:, None] * noise_a, axis=0)
        self.U_s += np.sum(weights[:, None] * noise_s, axis=0)
        
        self.U_a = np.clip(self.U_a, -d_max, a_max)
        self.U_s = np.clip(self.U_s, -s_max, s_max)
        
        # Extract first control action (receding horizon)
        cmd_a = self.U_a[0]
        cmd_s = self.U_s[0]
        
        # Shift nominal sequence forward
        self.U_a[:-1] = self.U_a[1:]
        self.U_a[-1] = 0.3  # Small forward bias to prevent stalling
        self.U_s[:-1] = self.U_s[1:]
        self.U_s[-1] = 0.0
        
        # Convert to Twist command
        cmd_v = float(np.clip(max(self.current_vel, 0.0) + cmd_a * dt, 0.0, v_max))
        
        # Publish TwistStamped → /cmd_vel_raw → AEB pipeline
        # IMPORTANT: angular.z = steering angle directly (NOT angular velocity)
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = cmd_v
        msg.twist.angular.z = float(cmd_s)
        self.cmd_pub.publish(msg)
        
        comp_time = (time.perf_counter() - t0) * 1000.0
        actual_freq = 1000.0 / comp_time if comp_time > 0 else 0
        self.get_logger().info(
            f"MPPI: {comp_time:.1f}ms | {actual_freq:.0f}Hz | "
            f"v={cmd_v:.2f} steer={cmd_s:.3f} a={cmd_a:.2f} | "
            f"Col:{num_collisions}/{N}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = MPPITracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
