#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformListener
import numpy as np
import heapq
import math
import time
from scipy.ndimage import distance_transform_edt
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        
        # Parámetros del nodo
        self.declare_parameter('inflation_radius_m', 0.40)
        self.declare_parameter('downsample_factor', 2)
        self.declare_parameter('path_resolution_m', 0.05)
        # Real car: track 0.22 m -> half width 0.11 m + 0.09 m margin
        self.declare_parameter('lethal_radius_m', 0.20)
        self.declare_parameter('soft_radius_m', 0.65)

        qos = QoSProfile(
            depth=1, 
            reliability=ReliabilityPolicy.RELIABLE, 
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.laps_sub = self.create_subscription(Int32, '/laps_cmd', self.laps_callback, 10)
        
        self.path_pub = self.create_publisher(Path, '/plan', 10)
        self.lap_comp_pub = self.create_publisher(Empty, '/lap_completed', 10) 
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.map_grid = None
        self.inflated_grid = None
        self.small_grid = None
        self.map_info = None

        self.dynamic_wall = None

        # Máquina de Estados y Telemetría
        self.pending_race_laps = 0
        self.race_active = False
        self.laps_remaining = 0
        self.start_pose = None
        self.finish_line_normal = None
        self.lap_state = "IDLE"

        # Reloj central del sistema (20 Hz)
        self.system_tick_timer = self.create_timer(0.05, self.system_tick_callback)

        self.get_logger().info("Planificador Maestro Topológico Inicializado. Escáner adaptativo en línea.")

    def map_callback(self, msg):
        self.map_info = msg.info
        raw_grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        
        processed_grid = np.where(raw_grid == -1, 100, raw_grid)
        obstacle_mask = processed_grid > 50
        distances = distance_transform_edt(~obstacle_mask) * msg.info.resolution
        
        lethal_radius_m = self.get_parameter('lethal_radius_m').value
        soft_radius_m = self.get_parameter('soft_radius_m').value
        
        costmap = np.zeros_like(processed_grid, dtype=np.float32)
        costmap[distances <= lethal_radius_m] = 100.0
        
        soft_mask = (distances > lethal_radius_m) & (distances < soft_radius_m)
        ratio = (soft_radius_m - distances[soft_mask]) / (soft_radius_m - lethal_radius_m)
        costmap[soft_mask] = 49.0 * ratio
        
        self.inflated_grid = costmap.astype(np.int32)
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
            
        self.get_logger().info(f"Mapa estático procesado y en caché.")

    def goal_callback(self, msg):
        self.get_logger().info("Calculando trayectoria manual...")
        rx, ry, ryaw = self.get_robot_pose()
        if rx is None:
            self.get_logger().warn("Odometría no disponible.")
            return
            
        self.dynamic_wall = None
        goal_w = (msg.pose.position.x, msg.pose.position.y)
        raw_path = self.calculate_segment((rx, ry), goal_w)
        
        if raw_path:
            processed = self.post_process_path(raw_path)
            self.publish_path(processed)

    def laps_callback(self, msg):
        laps = msg.data
        if laps <= 0:
            self.pending_race_laps = 0
            self.race_active = False
            return
            
        self.pending_race_laps = laps
        self.get_logger().info(f"Comando F1 recibido: {laps} vueltas. Sincronizando sensores...")

    def system_tick_callback(self):
        if self.pending_race_laps > 0 and not self.race_active:
            self.try_compile_race()

        if self.race_active:
            self.process_telemetry()

    def try_compile_race(self):
        """ Compila la carrera garantizando un punto objetivo válido """
        if self.small_grid is None:
            return 
            
        rx, ry, ryaw = self.get_robot_pose()
        if rx is None:
            return 
            
        self.get_logger().info("Sistemas de Navegación Online. Evaluando entorno seguro...")
        t0 = time.perf_counter()

        # 1. Buscador Adaptativo: Escanear hacia atrás hasta encontrar un punto libre (fuera de muros)
        gx, gy = None, None
        for dist in np.arange(2.0, 7.0, 0.2): # Escanea desde 2m hasta 7m hacia atrás
            tx = rx - dist * math.cos(ryaw)
            ty = ry - dist * math.sin(ryaw)
            mx, my = self.world_to_small_map(tx, ty)
            
            if mx is not None and 0 <= mx < self.small_grid.shape[1] and 0 <= my < self.small_grid.shape[0]:
                if self.small_grid[my, mx] < 50: # Es un área libre
                    gx, gy = tx, ty
                    break

        if gx is None:
            self.get_logger().error("Error Crítico: No hay asfalto libre detrás del robot. Ubique el chasis en un área despejada.")
            self.pending_race_laps = 0
            return

        dist_found = math.hypot(gx - rx, gy - ry)
        self.get_logger().info(f"Punto de cierre asimilado a {dist_found:.1f}m detrás del chasis. Procesando topología masiva (esto tomará un momento)...")

        # 2. Instalar el muro de contención dinámico (Caja Matemática)
        self.dynamic_wall = (rx, ry, ryaw)

        # 3. Calcular la vuelta completa (forzado a avanzar)
        path_one_lap = self.calculate_segment((rx, ry), (gx, gy))

        # 4. Desmantelar la caja de seguridad
        self.dynamic_wall = None

        if not path_one_lap:
            self.get_logger().error("Imposible resolver la pista. Asegúrese de que el circuito sea un anillo continuo y no tenga bloqueos físicos.")
            self.pending_race_laps = 0
            return

        # 5. Coser las coordenadas
        path_one_lap.append((rx, ry))

        # 6. Multiplicar vueltas
        full_trajectory = []
        for i in range(self.pending_race_laps):
            if i == 0:
                full_trajectory.extend(path_one_lap)
            else:
                full_trajectory.extend(path_one_lap[1:])

        planning_time_ms = (time.perf_counter() - t0) * 1000.0

        if full_trajectory:
            processed_path = self.post_process_path(full_trajectory)
            self.publish_path(processed_path)
            self.get_logger().info(f"ÉXITO: Pista completa asimilada en {planning_time_ms/1000.0:.2f} segundos.")
            
            # 7. Levantar Línea de Meta para Telemetría
            self.start_pose = (rx, ry)
            self.finish_line_normal = (math.cos(ryaw), math.sin(ryaw)) 
            self.laps_remaining = self.pending_race_laps
            
            self.pending_race_laps = 0
            self.race_active = True
            self.lap_state = "JUST_CROSSED" 
            
            self.get_logger().info("Telemetría en línea. Controlador MPPI asume autoridad de vuelo. Planificador inactivo.")

    def process_telemetry(self):
        rx, ry, _ = self.get_robot_pose()
        if rx is None:
            return

        dist = math.hypot(rx - self.start_pose[0], ry - self.start_pose[1])
        dx = rx - self.start_pose[0]
        dy = ry - self.start_pose[1]
        proj = dx * self.finish_line_normal[0] + dy * self.finish_line_normal[1]

        if self.lap_state == "JUST_CROSSED":
            if dist > 5.0:
                self.lap_state = "ARMED"
                self.get_logger().info("Sensor de Meta ARMADO. Esperando intersección de plano...")
                
        elif self.lap_state == "ARMED":
            if dist < 4.0 and proj > 0.0:
                self.laps_remaining -= 1
                self.lap_state = "JUST_CROSSED"
                
                self.get_logger().info(f"🏁 ¡VUELTA REGISTRADA! Restan: {self.laps_remaining} vueltas")
                
                msg = Empty()
                self.lap_comp_pub.publish(msg)

                if self.laps_remaining <= 0:
                    self.get_logger().info("🏆 ¡MISIÓN FINALIZADA! Sistema entra en reposo absoluto.")
                    self.race_active = False

    def calculate_segment(self, start_world, goal_world):
        start = self.world_to_small_map(start_world[0], start_world[1])
        goal = self.world_to_small_map(goal_world[0], goal_world[1])

        if start[0] is None or goal[0] is None:
            return None

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        
        found = False

        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                found = True
                break
                
            for neighbor in self.get_neighbors(current):
                step_cost = math.sqrt((current[0]-neighbor[0])**2 + (current[1]-neighbor[1])**2)
                penalty = self.small_grid[neighbor[1], neighbor[0]] * 0.1
                tentative_g_score = g_score[current] + step_cost + penalty
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.octile_distance(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
        if found:
            raw_path = []
            curr = goal
            while curr in came_from:
                raw_path.append(curr)
                curr = came_from[curr]
            raw_path.append(start)
            raw_path.reverse()
            return [self.small_map_to_world(p[0], p[1]) for p in raw_path]
            
        return None

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
            
        if self.dynamic_wall is not None:
            rx, ry, ryaw = self.dynamic_wall
            wx, wy = self.small_map_to_world(x, y)
            
            # Transformación perfecta a coordenadas cinemáticas locales del robot
            dx = wx - rx
            dy = wy - ry
            local_x = dx * math.cos(ryaw) + dy * math.sin(ryaw)
            local_y = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
            
            # Caja Geométrica de Contención: 
            # -1.8m a -0.5m detrás del eje, y 7m de ancho total (±3.5m)
            # Absolutamente sin sangrado hacia curvas paralelas
            if -1.8 <= local_x <= -0.5 and -3.5 <= local_y <= 3.5:
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

    def post_process_path(self, path):
        if len(path) < 3:
            return path
            
        simplified = [path[0]]
        for i in range(1, len(path)-1):
            p1, p2, p3 = simplified[-1], path[i], path[i+1]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            if abs(cross) > 1e-4:
                simplified.append(p2)
        simplified.append(path[-1])
        
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