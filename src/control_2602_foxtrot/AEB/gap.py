#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

class PrecisionFTGNode(Node):
    def __init__(self):
        super().__init__('precision_ftg_node')

        # =========================================================
        # SYSTEM CONFIGURATION SWITCH
        # =========================================================
        # True = Fast Track Mode (No obstacles, optimized for 3.0 m/s)
        # False = Obstacle Mode (Hairpins, high-density slaloms at 1.4 m/s)
        self.ENABLE_TRACK_MODE = True 

        self._configure_kinematics()

        # State Variables
        self.prev_target_angle = 0.0
        self.target_initialized = False
        self.ftg_enabled = False

        # ROS 2 Interfaces
        self.create_subscription(Bool, '/ftg/state', self.state_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_nav', 10)

        mode_str = "FAST TRACK" if self.ENABLE_TRACK_MODE else "OBSTACLE THREADING"
        self.get_logger().info(f"Precision FTG Online. Kinematics configured for: {mode_str}")

    def _configure_kinematics(self):
        """Loads physical thresholds and heuristic weights based on the active mode."""
        if self.ENABLE_TRACK_MODE:
            # ---------------- FAST TRACK PARAMETERS ----------------
            self.forward_velocity = 6.0          
            self.kp_steering = 2.0              
            self.max_target_angle = math.radians(70.0) 
            self.max_angular_velocity = 5.5      
            self.max_target_change = math.radians(8.0) 

            self.safe_radius = 0.25              
            self.max_decel = 7.0                 
            self.stop_margin = 0.10
            self.lateral_margin = 0.30
            self.emergency_front_dist = 0.40     
            self.caution_front_dist = 2.5        

            # Subroutine Heuristics
            self.lidar_horizon = 10.0
            self.fov_rad = math.radians(85.0)
            self.gap_depth_exp = 2.0
            self.gap_angle_penalty = 0.10
            self.speed_scrub_factor = 0.45
            
            # Depth Weighting Geometry [min, max, offset, scale]
            self.dw_params = [0.20, 0.85, 0.5, 2.0]
            
        else:
            # -------------- OBSTACLE MODE PARAMETERS --------------
            self.forward_velocity = 1.4
            self.kp_steering = 2.4               
            self.max_target_angle = math.radians(90.0) 
            self.max_angular_velocity = 1.8
            self.max_target_change = math.radians(25.0) 

            self.safe_radius = 0.18
            self.max_decel = 2.5
            self.stop_margin = 0.10
            self.lateral_margin = 0.20
            self.emergency_front_dist = 0.28
            self.caution_front_dist = 0.75

            # Subroutine Heuristics
            self.lidar_horizon = 4.0
            self.fov_rad = math.radians(90.0)
            self.gap_depth_exp = 1.2
            self.gap_angle_penalty = 0.15
            self.speed_scrub_factor = 0.70

            # Depth Weighting Geometry [min, max, offset, scale]
            self.dw_params = [0.0, 0.35, 0.4, 1.1]

    def state_callback(self, msg):
        if self.ftg_enabled and not msg.data:
            self.stop_vehicle()
        self.ftg_enabled = msg.data

    def scan_callback(self, msg):
        if not self.ftg_enabled: 
            return

        processed_ranges = self._preprocess_lidar(msg)
        processed_ranges, min_dist = self._apply_safety_bubble(processed_ranges, msg.angle_increment)
        
        best_gap = self._find_largest_gap(processed_ranges, msg.angle_min, msg.angle_increment)
        if not best_gap:
            self.get_logger().warn("No traversable gap detected. Halting.")
            return self.stop_vehicle()

        target_angle = self._calculate_steering_angle(best_gap, processed_ranges, msg.angle_min, msg.angle_increment)
        front_clearance = self._measure_front_clearance(msg)

        self.drive(target_angle, front_clearance, min_dist)

    # ---------------------------------------------------------
    # Algorithm Subroutines
    # ---------------------------------------------------------

    def _preprocess_lidar(self, msg):
        ranges = []
        for i, r in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if abs(angle) > self.fov_rad or r < 0.1:
                ranges.append(0.0)  
            elif math.isnan(r) or math.isinf(r) or r > self.lidar_horizon: 
                ranges.append(self.lidar_horizon)  
            else:
                ranges.append(r)
        return ranges

    def _apply_safety_bubble(self, ranges, angle_increment):
        valid_ranges = [(i, r) for i, r in enumerate(ranges) if r > 0.0]
        if not valid_ranges:
            return ranges, self.lidar_horizon

        min_idx, min_dist = min(valid_ranges, key=lambda x: x[1])

        if min_dist < self.lidar_horizon:
            dyn_radius = self.safe_radius + (0.05 * self.forward_velocity)
            bubble_angle = math.atan2(dyn_radius, max(0.1, min_dist))
            num_idx = int(bubble_angle / angle_increment)
            
            start_idx = max(0, min_idx - num_idx)
            end_idx = min(len(ranges), min_idx + num_idx + 1)
            for i in range(start_idx, end_idx):
                ranges[i] = 0.0
                
        return ranges, min_dist

    def _find_largest_gap(self, ranges, angle_min, angle_increment):
        gaps = []
        start = -1
        for i, r in enumerate(ranges):
            if r > 0.0:
                if start == -1: 
                    start = i
            elif start != -1:
                gaps.append((start, i - start))
                start = -1
                
        if start != -1: 
            gaps.append((start, len(ranges) - start))

        if not gaps:
            return None

        viable_gaps = [g for g in gaps if g[1] >= 5]
        if not viable_gaps:
            viable_gaps = gaps

        def gap_score(gap):
            g_start, g_len = gap
            g_slice = ranges[g_start : g_start + g_len]
            
            avg_depth = sum(g_slice) / max(1, len(g_slice))
            gap_center_idx = g_start + (g_len / 2.0)
            gap_center_angle = angle_min + gap_center_idx * angle_increment
            
            angle_factor = 1.0 - self.gap_angle_penalty * (abs(gap_center_angle) / self.max_target_angle)
            
            return g_len * (avg_depth ** self.gap_depth_exp) * angle_factor

        return max(viable_gaps, key=gap_score)

    def _calculate_steering_angle(self, best_gap, ranges, angle_min, angle_increment):
        max_gap_start, max_gap_len = best_gap
        gap_slice = ranges[max_gap_start : max_gap_start + max_gap_len]

        max_depth = max(gap_slice)
        deepest_indices = [i for i, r in enumerate(gap_slice) if r >= max_depth * 0.95]
        deepest_local_idx = deepest_indices[len(deepest_indices) // 2] if deepest_indices else max_gap_len // 2
        
        local_min_dist = min(gap_slice) if gap_slice else 0.1
        
        d_min, d_max, d_offset, d_scale = self.dw_params
        depth_weight = max(d_min, min(d_max, (local_min_dist - d_offset) / d_scale))
        center_weight = 1.0 - depth_weight
        
        target_idx = int(center_weight * (max_gap_start + max_gap_len // 2) + depth_weight * (max_gap_start + deepest_local_idx))
        target_angle = max(-self.max_target_angle, min(self.max_target_angle, angle_min + target_idx * angle_increment))

        if self.target_initialized:
            delta = target_angle - self.prev_target_angle
            target_angle = self.prev_target_angle + max(-self.max_target_change, min(self.max_target_change, delta))
        
        self.prev_target_angle = target_angle
        self.target_initialized = True
        return target_angle

    def _measure_front_clearance(self, msg):
        cone_rad = math.radians(20.0)
        front_dists = [r for i, r in enumerate(msg.ranges) 
                       if abs(msg.angle_min + i * msg.angle_increment) <= cone_rad and 0.1 < r < float('inf')]
        return min(front_dists, default=float('inf'))

    # ---------------------------------------------------------
    # Drive Control
    # ---------------------------------------------------------

    def drive(self, target_angle, front_dist, lat_dist):
        angular_vel = max(-self.max_angular_velocity, min(self.max_angular_velocity, target_angle * self.kp_steering))
        
        speed = self.forward_velocity * (1.0 - self.speed_scrub_factor * min(1.0, abs(target_angle) / self.max_target_angle))

        if math.isfinite(front_dist):
            if front_dist <= self.emergency_front_dist:
                speed = 0.0 
            elif front_dist < self.caution_front_dist:
                ratio = (front_dist - self.emergency_front_dist) / (self.caution_front_dist - self.emergency_front_dist)
                speed *= max(0.0, min(1.0, ratio))

        if lat_dist < self.lateral_margin:
            speed *= max(0.25, lat_dist / self.lateral_margin)
            
        if math.isfinite(front_dist):
            available_dist = max(0.0, front_dist - self.stop_margin)
            if available_dist < 1.0:
                speed = min(speed, math.sqrt(2.0 * self.max_decel * available_dist))

        speed = max(0.0, speed)
        
        self.cmd_pub.publish(self.make_command(speed, angular_vel))

    def make_command(self, linear_x, angular_z):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x, cmd.twist.angular.z = float(linear_x), float(angular_z)
        return cmd

    def stop_vehicle(self):
        self.cmd_pub.publish(self.make_command(0.0, 0.0))

def main(args=None):
    rclpy.init(args=args)
    node = PrecisionFTGNode()
    try: 
        rclpy.spin(node)
    except KeyboardInterrupt: 
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()