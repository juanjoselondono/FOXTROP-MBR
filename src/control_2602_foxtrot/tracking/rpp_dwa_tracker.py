#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import math
from rclpy.qos import qos_profile_sensor_data

class RPPDWATracker(Node):
    def __init__(self):
        super().__init__('rpp_dwa_tracker')
        
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/ekf/odometry', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_raw', 10)
        
        # Adaptive Lookahead params
        self.l0 = 0.5
        self.kv = 0.5
        self.l_min = 0.5
        self.l_max = 2.5
        
        # Speed scheduling params
        self.v_max = 5.0
        self.ay_max = 3.0
        self.epsilon = 1e-6
        
        # State
        self.global_path = None
        self.current_pose = None
        self.current_vel = 0.0
        self.latest_scan = None
        
        self.timer = self.create_timer(0.05, self.control_loop) # 20Hz
        self.get_logger().info("RPP+DWA Tracker initialized.")

    def path_callback(self, msg):
        self.global_path = msg.poses

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.current_vel = msg.twist.twist.linear.x

    def scan_callback(self, msg):
        self.latest_scan = msg

    def get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def calculate_lookahead(self):
        return max(self.l_min, min(self.l_max, self.l0 + self.kv * abs(self.current_vel)))

    def find_lookahead_point(self, ld):
        if not self.global_path or not self.current_pose:
            return None
            
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        
        closest_idx = 0
        min_dist = float('inf')
        for i, p in enumerate(self.global_path):
            d = math.hypot(p.pose.position.x - rx, p.pose.position.y - ry)
            if d < min_dist:
                min_dist = d
                closest_idx = i
                
        # Find first point beyond ld
        for i in range(closest_idx, len(self.global_path)):
            p = self.global_path[i]
            d = math.hypot(p.pose.position.x - rx, p.pose.position.y - ry)
            if d >= ld:
                return p.pose.position
                
        # If end of path is closer than ld, just return end
        return self.global_path[-1].pose.position if self.global_path else None

    def transform_to_local(self, target_point):
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        ryaw = self.get_yaw(self.current_pose.orientation)
        
        dx = target_point.x - rx
        dy = target_point.y - ry
        
        local_x = dx * math.cos(-ryaw) - dy * math.sin(-ryaw)
        local_y = dx * math.sin(-ryaw) + dy * math.cos(-ryaw)
        
        return local_x, local_y

    def score_trajectory_dwa(self, v, w, target_v, target_w):
        heading_cost = abs(w - target_w)
        vel_cost = abs(target_v - v)
        
        min_obs_dist = float('inf')
        if self.latest_scan:
            dt = 0.5
            px = v * dt
            py = 0
            if w != 0 and v != 0:
                R = v/w
                px = R * math.sin(w*dt)
                py = R * (1 - math.cos(w*dt))
            
            for i, r in enumerate(self.latest_scan.ranges):
                # Ignore readings < 0.4 to prevent self-collisions with the chassis!
                if math.isinf(r) or math.isnan(r) or r < 0.4:
                    continue
                angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment
                ox = r * math.cos(angle)
                oy = r * math.sin(angle)
                
                dist = math.hypot(px - ox, py - oy)
                if dist < min_obs_dist:
                    min_obs_dist = dist
                    
        # Collision threshold for actual obstacles (since we ignore < 0.4)
        if min_obs_dist < 0.3: 
            return float('inf')
            
        obs_cost = 1.0 / (min_obs_dist + 0.1)
        return 1.0 * heading_cost + 1.0 * vel_cost + 5.0 * obs_cost

    def control_loop(self):
        if not self.global_path:
            return
        if not self.current_pose:
            return
            
        ld = self.calculate_lookahead()
        target = self.find_lookahead_point(ld)
        
        if not target:
            return
            
        # Check if we reached the final goal
        final_goal = self.global_path[-1].pose.position
        dist_to_goal = math.hypot(final_goal.x - self.current_pose.position.x, final_goal.y - self.current_pose.position.y)
        if dist_to_goal < 0.25:
            # Reached goal! Stop.
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.0
            self.cmd_pub.publish(msg)
            return
            
        local_x, local_y = self.transform_to_local(target)
        
        dist_sq = local_x**2 + local_y**2
        
        # If target is behind the robot, just turn in place
        if local_x < 0.0:
            v_target = 0.0
            w_target = 1.0 if local_y > 0 else -1.0
        else:
            if dist_sq < 0.01:
                kappa = 0.0
            else:
                kappa = (2.0 * local_y) / dist_sq
            v_target = min(self.v_max, math.sqrt(self.ay_max / (abs(kappa) + self.epsilon)))
            w_target = v_target * kappa
        
        best_v = 0.0
        best_w = 0.0
        min_cost = float('inf')
        
        v_samples = np.linspace(max(0.0, self.current_vel - 1.0), min(self.v_max, self.current_vel + 1.0), 5)
        w_samples = np.linspace(w_target - 1.0, w_target + 1.0, 7)
        
        for v in v_samples:
            for w in w_samples:
                cost = self.score_trajectory_dwa(v, w, v_target, w_target)
                if cost < min_cost:
                    min_cost = cost
                    best_v = v
                    best_w = w
                    
        # If no valid trajectory found, AEB or DWA will stop
        if min_cost == float('inf'):
            best_v = 0.0
            best_w = 0.0
            
        # Publish
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(best_v)
        msg.twist.angular.z = float(best_w)
        
        # Log to debug
        self.get_logger().info(f"Publishing cmd_vel: v={best_v:.2f}, w={best_w:.2f}")
        
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RPPDWATracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

