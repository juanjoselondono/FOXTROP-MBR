#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import numpy as np
import random
import math

class RRTNodeObj:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRTPlanner(Node):
    def __init__(self):
        super().__init__('rrt_pp_2602_foxtrot')
        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cb, map_qos)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_cb, 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.declare_parameter('step_size', 0.15)
        self.declare_parameter('max_iter', 15000)
        self.declare_parameter('goal_bias', 0.15)
        self.declare_parameter('collision_threshold', 65)
        
        self.grid = None
        self.map_info = None

    def map_cb(self, msg):
        self.map_info = msg.info
        self.grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    def goal_cb(self, msg):
        self.get_logger().info('>>> OBJETIVO RECIBIDO DESDE RVIZ2 <<<')
        if self.grid is None:
            self.get_logger().error('CRÍTICO: Objetivo recibido, pero el mapa no ha cargado.')
            return
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            start_x = trans.transform.translation.x
            start_y = trans.transform.translation.y
        except Exception as e:
            self.get_logger().error(f'TF Error: {e}')
            return

        start_node = RRTNodeObj(start_x, start_y)
        goal_node = RRTNodeObj(msg.pose.position.x, msg.pose.position.y)
        
        path = self.plan(start_node, goal_node)
        if path:
            self.path_pub.publish(path)
            self.get_logger().info(f'Ruta generada: {len(path.poses)} waypoints.')

    def plan(self, start, goal):
        tree = [start]
        step_size = self.get_parameter('step_size').value
        goal_bias = self.get_parameter('goal_bias').value
        width_m = self.map_info.width * self.map_info.resolution
        height_m = self.map_info.height * self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        
        for _ in range(self.get_parameter('max_iter').value):
            if random.random() < goal_bias:
                rand_node = goal
            else:
                rx = random.uniform(origin_x, origin_x + width_m)
                ry = random.uniform(origin_y, origin_y + height_m)
                rand_node = RRTNodeObj(rx, ry)
                
            nearest_node = min(tree, key=lambda n: math.hypot(n.x - rand_node.x, n.y - rand_node.y))
            theta = math.atan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)
            new_node = RRTNodeObj(nearest_node.x + step_size * math.cos(theta), 
                                  nearest_node.y + step_size * math.sin(theta))
            new_node.parent = nearest_node
            
            if not self.check_collision(nearest_node, new_node):
                tree.append(new_node)
                if math.hypot(new_node.x - goal.x, new_node.y - goal.y) < step_size:
                    goal.parent = new_node
                    tree.append(goal)
                    return self.reconstruct_path(goal)
        return None

    def check_collision(self, n1, n2):
        steps = int(math.hypot(n2.x - n1.x, n2.y - n1.y) / self.map_info.resolution)
        threshold = self.get_parameter('collision_threshold').value
        
        # Buffer cinemático: 20 cm de radio (~4 celdas a 0.05m/celda)
        clearance_cells = int(0.20 / self.map_info.resolution) 
        
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = n1.x + t * (n2.x - n1.x)
            y = n1.y + t * (n2.y - n1.y)
            
            ix = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
            iy = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
            
            # Escaneo de perímetro en lugar de punto único
            for dx in range(-clearance_cells, clearance_cells + 1):
                for dy in range(-clearance_cells, clearance_cells + 1):
                    nx, ny = ix + dx, iy + dy
                    if 0 <= nx < self.map_info.width and 0 <= ny < self.map_info.height:
                        val = self.grid[ny][nx]
                        if val >= threshold or val == -1:
                            return True
                    else:
                        return True
        return False

    def reconstruct_path(self, node):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        curr = node
        while curr is not None:
            pose = PoseStamped()
            pose.header.frame_id = 'map' # FIX: ROS 2 requiere frame_id en cada elemento
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = float(curr.x)
            pose.pose.position.y = float(curr.y)
            pose.pose.orientation.w = 1.0 
            path_msg.poses.insert(0, pose)
            curr = curr.parent
        return path_msg

def main(args=None):
    rclpy.init(args=args)
    node = RRTPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()