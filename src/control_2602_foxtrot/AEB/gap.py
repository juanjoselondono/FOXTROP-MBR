#!/usr/bin/env python3
import rclpy, math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

class PrecisionFTGNode(Node):
    def __init__(self):
        super().__init__('precision_ftg_node')

        # Vehicle & Control Parameters
        self.forward_velocity = 2.2
        self.kp_steering = 1.8
        self.max_target_angle = math.radians(60.0)
        self.max_angular_velocity = 1.2
        self.max_target_change = math.radians(15.0)

        # Safety & Braking Parameters
        self.safe_radius = 0.405
        self.max_decel = 2.5
        self.stop_margin = 0.15
        self.lateral_margin = 0.45
        self.emergency_front_dist = 0.55
        self.caution_front_dist = 1.2

        # State Variables
        self.prev_target_angle = 0.0
        self.target_initialized = False
        self.ftg_enabled = False

        # ROS 2 Interfaces
        self.create_subscription(Bool, '/ftg/state', self.state_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_nav', 10)

        self.get_logger().info("Precision FTG Online. Forward-progress constrained.")

    def state_callback(self, msg):
        if self.ftg_enabled and not msg.data:
            self.stop_vehicle()
        self.ftg_enabled = msg.data

    def scan_callback(self, msg):
        if not self.ftg_enabled: return

        # 1. Forward FOV & Preprocessing
        fov_rad = math.radians(75.0)
        ranges = []
        for i, r in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if abs(angle) > fov_rad or r < 0.1:
                ranges.append(0.0)
            elif math.isnan(r) or math.isinf(r) or r > 4.0:
                ranges.append(4.0)
            else:
                ranges.append(r)

        # 2. Find Global Closest Obstacle
        valid_ranges = [(i, r) for i, r in enumerate(ranges) if r > 0.0]
        min_idx, min_dist = min(valid_ranges, key=lambda x: x[1], default=(-1, 4.0))

        # 3. Dynamic Safety Bubble
        if min_idx != -1 and min_dist < 4.0:
            dyn_radius = self.safe_radius + 0.10 * self.forward_velocity
            bubble_angle = math.asin(min(1.0, dyn_radius / max(0.1, min_dist)))
            num_idx = int(bubble_angle / msg.angle_increment)
            for i in range(max(0, min_idx - num_idx), min(len(ranges), min_idx + num_idx + 1)):
                ranges[i] = 0.0

        # 4. Find Largest Gap
        gaps, start = [], -1
        for i, r in enumerate(ranges):
            if r > 0.0:
                if start == -1: start = i
            elif start != -1:
                gaps.append((start, i - start))
                start = -1
        if start != -1: gaps.append((start, len(ranges) - start))

        if not gaps:
            self.get_logger().warn("No traversable gap. Stopping.")
            return self.stop_vehicle()

        max_gap_start, max_gap_len = max(gaps, key=lambda x: x[1])
        gap_slice = ranges[max_gap_start : max_gap_start + max_gap_len]

        # 5. Blend Gap Center with Deepest Point
        max_depth = max(gap_slice)
        deepest_indices = [i for i, r in enumerate(gap_slice) if r >= max_depth * 0.95]
        deepest_local_idx = deepest_indices[len(deepest_indices) // 2] if deepest_indices else max_gap_len // 2
        
        target_idx = int(0.65 * (max_gap_start + max_gap_len // 2) + 0.35 * (max_gap_start + deepest_local_idx))
        target_angle = max(-self.max_target_angle, min(self.max_target_angle, msg.angle_min + target_idx * msg.angle_increment))

        # 6. Target Smoothing
        if self.target_initialized:
            delta = target_angle - self.prev_target_angle
            target_angle = self.prev_target_angle + max(-self.max_target_change, min(self.max_target_change, delta))
        
        self.prev_target_angle = target_angle
        self.target_initialized = True

        # 7. Front Distance Calculation (20-degree cone)
        cone_rad = math.radians(20.0)
        front_dists = [r for i, r in enumerate(msg.ranges) if abs(msg.angle_min + i * msg.angle_increment) <= cone_rad and 0.1 < r < float('inf')]
        min_front_dist = min(front_dists, default=float('inf'))

        # 8. Drive Command Execution
        self.drive(target_angle, min_front_dist, min_dist)

    def drive(self, target_angle, front_dist, lat_dist):
        # Base Control
        angular_vel = max(-self.max_angular_velocity, min(self.max_angular_velocity, target_angle * self.kp_steering))
        speed = self.forward_velocity * (1.0 - 0.60 * min(1.0, abs(target_angle) / self.max_target_angle))

        # Front Braking Pipeline
        if math.isfinite(front_dist):
            if front_dist <= self.emergency_front_dist:
                speed, angular_vel = 0.0, 0.0
            elif front_dist < self.caution_front_dist:
                ratio = (front_dist - self.emergency_front_dist) / (self.caution_front_dist - self.emergency_front_dist)
                speed *= max(0.0, min(1.0, ratio))

        # Lateral & Kinematic Braking
        if lat_dist < self.lateral_margin:
            speed *= max(0.25, lat_dist / self.lateral_margin)
            
        if math.isfinite(front_dist):
            available_dist = max(0.0, front_dist - self.stop_margin)
            if available_dist < 1.0:
                speed = min(speed, math.sqrt(2.0 * self.max_decel * available_dist))

        # Final Safety Verification
        speed = max(0.0, speed)
        if speed <= 0.01:
            angular_vel = 0.0

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
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()