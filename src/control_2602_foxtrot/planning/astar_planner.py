#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import numpy as np
import heapq
import math
import time
from scipy.ndimage import binary_dilation, generate_binary_structure
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        
        # Parameters
        self.declare_parameter('inflation_radius_m', 0.45)
        self.declare_parameter('downsample_factor', 2)
        self.declare_parameter('path_resolution_m', 0.05)
        
        qos = QoSProfile(
            depth=1, 
            reliability=ReliabilityPolicy.RELIABLE, 
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        from std_msgs.msg import Int32
        self.laps_sub = self.create_subscription(Int32, '/laps_cmd', self.laps_callback, 10)
        self.path_pub = self.create_publisher(Path, '/plan', 10) # Keeping /plan as it matches RViz config
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.map_grid = None
        self.inflated_grid = None
        self.small_grid = None
        self.map_info = None
        
        self.timer = self.create_timer(1.0, self.plan_timer_callback)
        self.current_goal = None

    def map_callback(self, msg):
        self.map_info = msg.info
        raw_grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        
        # Treat unknown (-1) conservatively as obstacles (100)
        processed_grid = np.where(raw_grid == -1, 100, raw_grid)
        
        inflation_radius_m = self.get_parameter('inflation_radius_m').value
        inflation_cells = math.ceil(inflation_radius_m / msg.info.resolution)
        
        # Inflate obstacles using binary_dilation
        obstacle_mask = processed_grid > 50
        struct = generate_binary_structure(2, 2) # 8-connected
        inflated_mask = binary_dilation(obstacle_mask, structure=struct, iterations=inflation_cells)
        
        self.inflated_grid = np.where(inflated_mask, 100, 0)
        
        # F1 MODE: Break the track loop with a virtual wall BEHIND the start line
        # Map frame is rotated ~180° from Gazebo. Robot at map (-0.026, -0.581) faces -X.
        # "Behind" the robot (backward direction) is +X in map frame.
        # Wall blocks the backward shortcut so A* must plan forward around the full loop.
        f1_mode = True
        if f1_mode:
            wall_x_m = 1.5    # +X = behind the robot in map frame
            wall_y_m = -0.6   # Same Y as the robot's start position
            wall_radius_m = 0.8
            
            w_gx = int((wall_x_m - msg.info.origin.position.x) / msg.info.resolution)
            w_gy = int((wall_y_m - msg.info.origin.position.y) / msg.info.resolution)
            w_r = int(wall_radius_m / msg.info.resolution)
            
            for y in range(max(0, w_gy - w_r), min(msg.info.height, w_gy + w_r)):
                for x in range(max(0, w_gx - w_r), min(msg.info.width, w_gx + w_r)):
                    if (x - w_gx)**2 + (y - w_gy)**2 <= w_r**2:
                        self.inflated_grid[y, x] = 100
            
            self.get_logger().info(f"F1 Virtual Wall placed at map ({wall_x_m}, {wall_y_m}), radius {wall_radius_m}m, grid ({w_gx}, {w_gy})")
        
        # Downsample preserving obstacles (Max Pooling)
        self.downsample_factor = self.get_parameter('downsample_factor').value
        if self.downsample_factor > 1:
            h, w = self.inflated_grid.shape
            k = self.downsample_factor
            new_h, new_w = h // k, w // k
            trimmed = self.inflated_grid[:new_h*k, :new_w*k]
            reshaped = trimmed.reshape(new_h, k, new_w, k)
            self.small_grid = reshaped.max(axis=(1, 3))
        else:
            self.small_grid = self.inflated_grid
            
        self.get_logger().info(f"Map processed: inflated by {inflation_cells} cells ({inflation_radius_m}m), downsampled by {self.downsample_factor}")

    def laps_callback(self, msg):
        self.get_logger().info(f"Received Laps Command: {msg.data}. Setting goal behind virtual wall.")
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        # Goal at map (+2.5, -0.6): behind the virtual wall at (+1.5, -0.6)
        # The robot at (-0.026, -0.581) must go FORWARD (-X) around the entire loop
        # to reach this point from the other direction.
        goal.pose.position.x = 2.5
        goal.pose.position.y = -0.6
        goal.pose.orientation.w = 1.0
        self.current_goal = goal
        self.plan_path()

    def goal_callback(self, msg):
        self.current_goal = msg
        self.plan_path()

    def plan_timer_callback(self):
        if self.current_goal is not None and self.small_grid is not None:
            self.plan_path()

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            q = trans.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return trans.transform.translation.x, trans.transform.translation.y, yaw
        except Exception:
            return None, None, None

    def world_to_small_map(self, wx, wy):
        if not self.map_info: return None, None
        mx = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
        my = int((wy - self.map_info.origin.position.y) / self.map_info.resolution)
        return mx // self.downsample_factor, my // self.downsample_factor

    def small_map_to_world(self, sx, sy):
        mx = sx * self.downsample_factor + (self.downsample_factor / 2.0)
        my = sy * self.downsample_factor + (self.downsample_factor / 2.0)
        wx = (mx * self.map_info.resolution) + self.map_info.origin.position.x
        wy = (my * self.map_info.resolution) + self.map_info.origin.position.y
        return wx, wy

    def is_valid_small(self, x, y):
        if x < 0 or x >= self.small_grid.shape[1] or y < 0 or y >= self.small_grid.shape[0]:
            return False
        return self.small_grid[y, x] < 50

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid_small(nx, ny):
                if dx != 0 and dy != 0:
                    if not self.is_valid_small(x + dx, y) or not self.is_valid_small(x, y + dy):
                        continue
                neighbors.append((nx, ny))
        return neighbors

    def octile_distance(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def plan_path(self):
        if self.small_grid is None or self.current_goal is None:
            return

        rx, ry, ryaw = self.get_robot_pose()
        if rx is None:
            return

        start = self.world_to_small_map(rx, ry)
        goal = self.world_to_small_map(self.current_goal.pose.position.x, self.current_goal.pose.position.y)

        if start[0] is None or goal[0] is None:
            return

        t0 = time.perf_counter()
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        
        expanded_nodes = 0
        found = False
        
        dir_x = math.cos(ryaw)
        dir_y = math.sin(ryaw)

        while open_set:
            current_f, current = heapq.heappop(open_set)
            expanded_nodes += 1
            
            if current == goal:
                found = True
                break
                
            for neighbor in self.get_neighbors(current):
                step_cost = math.sqrt((current[0]-neighbor[0])**2 + (current[1]-neighbor[1])**2)
                
                # F1 constraint: Penalize backward movement within 40 cells (~4m) of start
                dist_from_start = math.sqrt((current[0] - start[0])**2 + (current[1] - start[1])**2)
                if dist_from_start < 40:
                    vec_x = neighbor[0] - current[0]
                    vec_y = neighbor[1] - current[1]
                    dot = (vec_x * dir_x) + (vec_y * dir_y)
                    if dot < 0:
                        step_cost += 5000.0
                
                tentative_g_score = g_score[current] + step_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.octile_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
        planning_time_ms = (time.perf_counter() - t0) * 1000.0

        if found:
            raw_path = []
            curr = goal
            while curr in came_from:
                raw_path.append(curr)
                curr = came_from[curr]
            raw_path.append(start)
            raw_path.reverse()
            
            world_path = [self.small_map_to_world(p[0], p[1]) for p in raw_path]
            processed_path = self.post_process_path(world_path)
            
            path_length_m = sum(math.hypot(processed_path[i][0]-processed_path[i-1][0], 
                                           processed_path[i][1]-processed_path[i-1][1]) for i in range(1, len(processed_path)))
            
            self.get_logger().info(f"A* Success | Time: {planning_time_ms:.2f} ms | Nodes: {expanded_nodes} | Length: {path_length_m:.2f} m | Waypoints: {len(processed_path)}")
            self.publish_path(processed_path)
        else:
            self.get_logger().warn(f"A* Failed | Time: {planning_time_ms:.2f} ms | Nodes: {expanded_nodes}")

    def post_process_path(self, path):
        if len(path) < 3:
            return path
            
        # 1. Remove redundant collinear points
        simplified = [path[0]]
        for i in range(1, len(path)-1):
            p1, p2, p3 = simplified[-1], path[i], path[i+1]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            if abs(cross) > 1e-4:
                simplified.append(p2)
        simplified.append(path[-1])
        
        # 2. Simple shortcut smoothing (collision-aware)
        smoothed = [simplified[0]]
        for i in range(1, len(simplified)-1):
            p0 = smoothed[-1]
            p1 = simplified[i]
            p2 = simplified[i+1]
            
            new_x = 0.5 * p0[0] + 0.5 * p1[0]
            new_y = 0.5 * p0[1] + 0.5 * p1[1]
            
            mx = int((new_x - self.map_info.origin.position.x) / self.map_info.resolution)
            my = int((new_y - self.map_info.origin.position.y) / self.map_info.resolution)
            if 0 <= mx < self.inflated_grid.shape[1] and 0 <= my < self.inflated_grid.shape[0] and self.inflated_grid[my, mx] < 50:
                smoothed.append((new_x, new_y))
            else:
                smoothed.append(p1)
        smoothed.append(simplified[-1])
        
        # 3. Interpolation
        res = self.get_parameter('path_resolution_m').value
        interpolated = []
        for i in range(len(smoothed)-1):
            p1, p2 = smoothed[i], smoothed[i+1]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            num_pts = max(1, int(dist / res))
            for j in range(num_pts):
                interpolated.append((
                    p1[0] + (p2[0]-p1[0]) * j / num_pts,
                    p1[1] + (p2[1]-p1[1]) * j / num_pts
                ))
        interpolated.append(smoothed[-1])
        return interpolated

    def publish_path(self, world_path):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for p in world_path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            msg.poses.append(pose)
        self.path_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
