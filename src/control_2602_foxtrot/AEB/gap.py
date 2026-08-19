#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

class KinematicConfig:
    """Struct to hold logically grouped threshold and heuristic parameters."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class PrecisionFTGNode(Node):
    def __init__(self):
        super().__init__('precision_ftg_node')
        #TRACK Mode
        self.cfg = KinematicConfig(
            v_max=6.0, kp_steer=1.8, max_steer_rad=math.radians(70.0),
            max_omega=5.5, max_steer_delta=math.radians(8.0),
            safe_rad=0.25, max_decel=7.0, stop_margin=0.10, lat_margin=0.30,
            dist_emerg=0.40, dist_warn=2.5,
            lidar_horizon=10.0, fov=math.radians(85.0),
            gap_exp=2.0, gap_pen=0.10, scrub=0.45,
            dw_params=[0.20, 0.85, 0.5, 2.0] # Depth Weighting [min, max, offset, scale]
        )
        #Obstacle Avoidance Mode
        # self.cfg = KinematicConfig(
        #     v_max=3.5, kp_steer=5.5, max_steer_rad=math.radians(90.0),
        #     max_omega=4.0, max_steer_delta=math.radians(30.0),
        #     safe_rad=0.2, max_decel=3.5, stop_margin=0.10, lat_margin=0.22,
        #     dist_emerg=0.3, dist_warn=0.75,
        #     lidar_horizon=4.0, fov=math.radians(90.0),
        #     gap_exp=1.2, gap_pen=0.15, scrub=0.70,
        #     dw_params=[0.0, 0.35, 0.4, 1.1] # Depth Weighting [min, max, offset, scale]
        #     #once the winning gap is chosen, the robot must decide whether to aim at the geometric center of the gap or the deepest point of the gap.
        # )
        # State Variables
        self.prev_angle = 0.0
        self.target_initialized = False
        self.ftg_enabled = False

        # ROS 2 Interfaces
        self.create_subscription(Bool, '/ftg/state', self.state_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_nav', 10)
        self.get_logger().info(f"Precision FTG Online. Kinematics configured")

    def state_callback(self, msg):
        if self.ftg_enabled and not msg.data:
            self._publish_cmd(0.0, 0.0)
        self.ftg_enabled = msg.data

    def scan_callback(self, msg):
        if not self.ftg_enabled: 
            return

        # 1. GETS LIDAR DATA AND PROCESSES IT   
        ranges, closest_dist = self._process_lidar(msg)
        front_dist = self._get_front_clearance(msg)
        
        # 2. DECISION
        best_gap = self._find_best_gap(ranges, msg.angle_min, msg.angle_increment)
        if not best_gap:
            self.get_logger().warn("No traversable gap detected. Stopping.")
            return self._publish_cmd(0.0, 0.0)

        target_angle = self._calculate_steering(best_gap, ranges, msg.angle_min, msg.angle_increment)

        # 3. CONTROL
        self._execute_drive_control(target_angle, front_dist, closest_dist)

    # ---------------------------------------------------------
    # 1. PERCEPTION GEOMETRY
    # ---------------------------------------------------------

    def _process_lidar(self, msg):
        """Single-pass filter for FOV limits, bad data, and dynamic safety bubble."""
        ranges = list(msg.ranges)
        min_dist = float('inf')
        min_idx = -1

        # Apply FOV and range constraints, while tracking the closest valid obstacle
        for i, r in enumerate(ranges):
            angle = abs(msg.angle_min + i * msg.angle_increment)
            if angle > self.cfg.fov or r < 0.1:
                ranges[i] = 0.0  
            elif math.isnan(r) or math.isinf(r) or r > self.cfg.lidar_horizon: 
                ranges[i] = self.cfg.lidar_horizon  
            else:
                if r < min_dist:
                    min_dist, min_idx = r, i

        # SAFETY BUBBLE
        # Find the closest obstacle, then artificially mark a region around it as blocked so the robot cannot choose a trajectory that passes too close to it.
        # RADIO DINAMICO EN FUNCION DE LA VELOCIDAD DEL VEHICULO y la distancia minima de seguridad
        if min_dist < self.cfg.lidar_horizon and min_idx != -1:
            dyn_radius = self.cfg.safe_rad + (0.05 * self.cfg.v_max)
            bubble_angle = math.atan2(dyn_radius, max(0.1, min_dist))
            num_idx = int(bubble_angle / msg.angle_increment)
            
            start = max(0, min_idx - num_idx)
            end = min(len(ranges), min_idx + num_idx + 1)
            for i in range(start, end):
                ranges[i] = 0.0
                
        return ranges, min_dist

    def _get_front_clearance(self, msg):
        """Measures forward clearance using a highly efficient generator."""
        # Si hay un obstaculo dentro del cono frontal de 20 grados, devuelve la distancia mínima a ese obstáculo.
        cone_rad = math.radians(20.0)
        return min((r for i, r in enumerate(msg.ranges) 
                    if 0.1 < r < float('inf') and abs(msg.angle_min + i * msg.angle_increment) <= cone_rad), 
                   default=float('inf'))

    # ---------------------------------------------------------
    # 2. DECISION LOGIC
    # ---------------------------------------------------------

    def _find_best_gap(self, ranges, angle_min, angle_increment):
        """Extracts continuous gaps and selects the most traversable option."""
        gaps, start = [], -1
        # we scan indexes of the ranges array Whenever it sees a number greater than 0.0, it thinks, "Ah, here is the start of an opening."
        for i, r in enumerate(ranges):
            if r > 0.0:
                if start == -1: start = i
            elif start != -1:
                gaps.append((start, i - start))
                start = -1
        if start != -1: gaps.append((start, len(ranges) - start))

        if not gaps: return None

        # Filter out noise (gaps smaller than 5 rays)
        viable_gaps = [g for g in gaps if g[1] >= 5] or gaps
        # Score each viable gap using three factors:
        # 1. Width: Wider gaps are safer and score higher.
        # 2. Depth: Deeper gaps are strongly preferred because they lead farther ahead.
        # 3. Straightness: Forward-facing gaps score higher to avoid sharp steering.
        def score_gap(gap):
            g_start, g_len = gap
            g_slice = ranges[g_start : g_start + g_len]
            avg_depth = sum(g_slice) / max(1, g_len)
            
            center_angle = angle_min + (g_start + (g_len / 2.0)) * angle_increment
            # Penalize gaps that require sharp steering angles
            angle_factor = 1.0 - self.cfg.gap_pen * (abs(center_angle) / self.cfg.max_steer_rad)
            
            return g_len * (avg_depth ** self.cfg.gap_exp) * angle_factor

        return max(viable_gaps, key=score_gap)

    def _calculate_steering(self, best_gap, ranges, angle_min, angle_increment):
        """Determines the optimal steering angle within the chosen gap."""
        start_idx, length = best_gap
        gap_slice = ranges[start_idx : start_idx + length]

        # Find the deepest part of the gap to pull trajectory away from inner walls
        max_depth = max(gap_slice)
        deep_idx_list = [i for i, r in enumerate(gap_slice) if r >= max_depth * 0.95]
        local_deep_idx = deep_idx_list[len(deep_idx_list) // 2] if deep_idx_list else length // 2
        
        # Dynamic weighting: Wide gaps aim for center, narrow gaps aim for maximum depth
        d_min, d_max, d_offset, d_scale = self.cfg.dw_params
        local_min = min(gap_slice) if gap_slice else 0.1
        depth_weight = max(d_min, min(d_max, (local_min - d_offset) / d_scale))
        center_weight = 1.0 - depth_weight
        
        target_idx = int(center_weight * (length // 2) + depth_weight * local_deep_idx) + start_idx
        raw_angle = angle_min + target_idx * angle_increment
        clamped_angle = max(-self.cfg.max_steer_rad, min(self.cfg.max_steer_rad, raw_angle))

        # Enforce kinematic steering limits to prevent violent oscillation
        if self.target_initialized:
            delta = clamped_angle - self.prev_angle
            clamped_angle = self.prev_angle + max(-self.cfg.max_steer_delta, min(self.cfg.max_steer_delta, delta))
        
        self.prev_angle = clamped_angle
        self.target_initialized = True
        return clamped_angle

    # ---------------------------------------------------------
    # 3. CONTROL EXECUTION
    # ---------------------------------------------------------

    def _execute_drive_control(self, target_angle, front_dist, lat_dist):
        """Translates geometric targets into final velocity commands."""
        # Proportional steering control
        omega = max(-self.cfg.max_omega, min(self.cfg.max_omega, target_angle * self.cfg.kp_steer))
        
        # Base speed: scrubs velocity based on how sharp the turn is
        speed = self.cfg.v_max * (1.0 - self.cfg.scrub * min(1.0, abs(target_angle) / self.cfg.max_steer_rad))

        if math.isfinite(front_dist):
            # Forward proximity override
            if front_dist <= self.cfg.dist_emerg:
                speed = 0.0 
            elif front_dist < self.cfg.dist_warn:
                ratio = (front_dist - self.cfg.dist_emerg) / (self.cfg.dist_warn - self.cfg.dist_emerg)
                speed *= max(0.0, min(1.0, ratio))
            
            # Kinematic braking limit (v_max = sqrt(2 * a * d))
            avail_dist = max(0.0, front_dist - self.cfg.stop_margin)
            if avail_dist < 1.0:
                speed = min(speed, math.sqrt(2.0 * self.cfg.max_decel * avail_dist))

        # Lateral proximity override (slow down if walls are too close laterally)
        if lat_dist < self.cfg.lat_margin:
            speed *= max(0.25, lat_dist / self.cfg.lat_margin)
            
        self._publish_cmd(max(0.0, speed), omega)

    def _publish_cmd(self, linear_x, angular_z):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x, cmd.twist.angular.z = float(linear_x), float(angular_z)
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PrecisionFTGNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()