#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from tf2_ros import Buffer, TransformListener
import numpy as np
import heapq
import math
import time
from scipy.ndimage import binary_dilation, generate_binary_structure, distance_transform_edt
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        
        # Node parameters
        self.declare_parameter('inflation_radius_m', 0.40) # Increased to take wider corners
        self.declare_parameter('downsample_factor', 2)
        self.declare_parameter('path_resolution_m', 0.05)
        self.declare_parameter('goal_x', 2.5)
        self.declare_parameter('goal_y', -0.6)
        
        qos = QoSProfile(
            depth=1, 
            reliability=ReliabilityPolicy.RELIABLE, 
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.laps_sub = self.create_subscription(Int32, '/laps_cmd', self.laps_callback, 10)
        from std_msgs.msg import Empty
        self.lap_comp_sub = self.create_subscription(Empty, '/lap_completed', self.lap_completed_callback, 10)
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
        
        # Calculate Euclidean distance from every free cell to the nearest obstacle
        obstacle_mask = processed_grid > 50
        distances = distance_transform_edt(~obstacle_mask) * msg.info.resolution
        
        # Hard lethal radius: Absolute minimum clearance (robot width / 2)
        lethal_radius_m = 0.22 
        # Soft cost radius: Prefer to stay this far away from walls (inflation radius)
        soft_radius_m = 0.65 
        
        costmap = np.zeros_like(processed_grid, dtype=np.float32)
        costmap[distances <= lethal_radius_m] = 100.0
        
        # Soft cost gradient from 49 (near lethal) to 0 (at soft radius)
        # We cap at 49 because `is_valid_small` treats >= 50 as fatal
        soft_mask = (distances > lethal_radius_m) & (distances < soft_radius_m)
        ratio = (soft_radius_m - distances[soft_mask]) / (soft_radius_m - lethal_radius_m)
        costmap[soft_mask] = 49.0 * ratio
        
        self.inflated_grid = costmap.astype(np.int32)
        
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
            
        self.get_logger().info(f"Map processed: continuous costmap generated, downsampled by {self.downsample_factor}")



    def goal_callback(self, msg):
        self.get_logger().info("Received Manual Navigation Goal from RViz")
        self.dynamic_wall = None  # Clear race wall for manual planning
        self.current_goal = msg
        self.plan_path()

    def plan_timer_callback(self):
        pass # Disabled for F1 Racing Mode

    def lap_completed_callback(self, msg):
        self.get_logger().info("Lap completed! Generating next loop.")
        self.plan_path()

    def laps_callback(self, msg):
        self.laps_remaining = msg.data
        if self.laps_remaining <= 0:
            return
            
        # Get robot pose
        rx, ry, ryaw = self.get_robot_pose()
        if rx is None:
            self.get_logger().error("Cannot plan F1 lap: Robot pose unknown.")
            return

        # Auto-calculate the F1 Goal: Search behind the dynamic wall (-2.6m to -5.0m)
        # to find a valid free cell, so A* never fails trying to reach an obstacle.
        valid_goal_found = False
        goal_x, goal_y = rx, ry
        
        # We must set self.dynamic_wall temporarily to use is_valid_small
        self.dynamic_wall = (rx, ry, ryaw)
        
        # Fan search: search backwards in an arc to find a valid free cell, even on curves
        for dist in np.arange(2.6, 6.0, 0.2):
            for angle_offset in [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2]:
                test_yaw = ryaw + angle_offset
                test_gx = rx - dist * math.cos(test_yaw)
                test_gy = ry - dist * math.sin(test_yaw)
                
                # Check if this cell is physically free
                gx_idx, gy_idx = self.world_to_small_map(test_gx, test_gy)
                if gx_idx is not None and gy_idx is not None:
                    # Allow cells that are not completely lethal
                    if 0 <= gy_idx < self.small_grid.shape[0] and 0 <= gx_idx < self.small_grid.shape[1]:
                        if self.small_grid[gy_idx, gx_idx] < 50:
                            goal_x, goal_y = test_gx, test_gy
                            valid_goal_found = True
                            break
            if valid_goal_found:
                break
                    
        if not valid_goal_found:
            self.get_logger().error("A* F1: Could not find a valid free cell behind the robot for the goal!")
            return

        self.get_logger().info(f"F1 Mode: {self.laps_remaining} laps. Goal auto-set behind robot at ({goal_x:.2f}, {goal_y:.2f}).")

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = float(goal_x)
        goal.pose.position.y = float(goal_y)
        goal.pose.orientation.w = 1.0
        self.current_goal = goal
        
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
        if self.small_grid[y, x] >= 50:
            return False
            
        # DYNAMIC VIRTUAL WALL (Guillotine Block)
        # Block a rectangle: 0.5m to 2.5m behind the robot, 8 meters wide (-4 to +4)
        if hasattr(self, 'dynamic_wall') and self.dynamic_wall is not None:
            rx, ry, ryaw = self.dynamic_wall
            wx, wy = self.small_map_to_world(x, y)
            
            # Convert map coordinates to robot's local frame
            dx = wx - rx
            dy = wy - ry
            local_x = dx * math.cos(-ryaw) - dy * math.sin(-ryaw)
            local_y = dx * math.sin(-ryaw) + dy * math.cos(-ryaw)
            
            if -2.5 <= local_x <= -0.5 and -4.0 <= local_y <= 4.0:
                return False
                
        return True

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
            
        self.dynamic_wall = (rx, ry, ryaw)
        
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

        while open_set:
            current_f, current = heapq.heappop(open_set)
            expanded_nodes += 1
            
            if current == goal:
                found = True
                break
                
            for neighbor in self.get_neighbors(current):
                step_cost = math.sqrt((current[0]-neighbor[0])**2 + (current[1]-neighbor[1])**2)
                
                # Add obstacle penalty cost
                # Multiply by 0.1 so a max soft cost of 49 equals 4.9 "meters" of extra distance
                penalty = self.small_grid[neighbor[1], neighbor[0]] * 0.1
                
                tentative_g_score = g_score[current] + step_cost + penalty
                
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
            
        # 1. Remove redundant collinear points to compress path definition
        simplified = [path[0]]
        for i in range(1, len(path)-1):
            p1, p2, p3 = simplified[-1], path[i], path[i+1]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            if abs(cross) > 1e-4:
                simplified.append(p2)
        simplified.append(path[-1])
        
        # 2. Interpolation along the exact safe grid segments
        res = self.get_parameter('path_resolution_m').value
        interpolated = []
        for i in range(len(simplified)-1):
            p1, p2 = simplified[i], simplified[i+1]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            num_pts = max(1, int(dist / res))
            for j in range(num_pts):
                interpolated.append((
                    p1[0] + (p2[0]-p1[0]) * j / num_pts,
                    p1[1] + (p2[1]-p1[1]) * j / num_pts
                ))
        interpolated.append(simplified[-1])
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
