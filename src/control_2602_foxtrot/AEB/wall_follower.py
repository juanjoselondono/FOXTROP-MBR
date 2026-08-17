#!/usr/bin/env python3
import rclpy, math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

class OptimizedWallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower_node')
        
        # PD Control & Geometric Parameters
        self.traj_d = 0.5       
        self.lookahead_L = 0.5    
        self.theta_deg = 45.0     
        self.kp = 1.8  # Slightly increased for snappier cornering
        self.kd = 0.15            
        
        # Kinematic & Safety Limits (FTG Integration)
        self.max_velocity = 1.0
        self.max_angular_vel = 1.8
        self.max_ray_range = 3.0  # Virtual wall limit
        self.cornering_repulsion_gain = 2.5
        
        # State Variables
        self.prev_error = 0.0
        self.last_time = 0.0
        self.wf_enabled = False
        
        # ROS 2 Interfaces
        self.create_subscription(Bool, '/wf/state', self.state_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_nav', 10)
        
        self.get_logger().info("Optimized PD Wall Follower online. FTG Kinematics engaged.")

    def state_callback(self, msg):
        if self.wf_enabled and not msg.data:
            self.stop_vehicle()
        self.wf_enabled = msg.data

    def get_clamped_range(self, msg, angle_deg):
        """ Returns the ray distance, clamped to a maximum range to prevent 'inf' math crashes. """
        angle_rad = math.radians(angle_deg)
        if angle_rad < msg.angle_min or angle_rad > msg.angle_max: 
            return self.max_ray_range
            
        idx = int((angle_rad - msg.angle_min) / msg.angle_increment)
        r = msg.ranges[idx]
        
        if math.isinf(r) or math.isnan(r) or r > self.max_ray_range:
            return self.max_ray_range
        return max(0.1, r)

    def scan_callback(self, msg):
        if not self.wf_enabled:
            self.last_time = 0.0
            return

        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_time == 0.0:
            self.last_time = current_time
            return
            
        dt = current_time - self.last_time
        if dt <= 0.0: return

        # 1. Frontal Obstacle Detection (Inward Corners)
        cone_rad = math.radians(30.0)
        front_dists = [r for i, r in enumerate(msg.ranges) 
                       if abs(msg.angle_min + i * msg.angle_increment) <= cone_rad and 0.1 < r < 4.0]
        min_front_dist = min(front_dists, default=4.0)

        # 2. Geometric Right-Wall Tracking (Clamped)
        b = self.get_clamped_range(msg, -90.0)
        a = self.get_clamped_range(msg, -90.0 + self.theta_deg)

        theta_rad = math.radians(self.theta_deg)
        denominator = a * math.sin(theta_rad)
        alpha = math.atan((a * math.cos(theta_rad) - b) / denominator) if abs(denominator) > 1e-6 else 0.0

        future_distance = (b * math.cos(alpha)) + (self.lookahead_L * math.sin(alpha))
        
        # 3. PD Control
        error = self.traj_d - future_distance
        error_dot = (error - self.prev_error) / dt
        self.prev_error = error
        self.last_time = current_time
        
        angular_vel = (self.kp * error) + (self.kd * error_dot)

        # 4. Inward Corner Repulsion Override
        # If a wall appears in front, inject a strong left turn to prevent collision
        if min_front_dist < 0.9:
            repulsion = (0.9 - min_front_dist) * self.cornering_repulsion_gain
            angular_vel += repulsion  # Positive rad/s drives left

        # 5. Kinematic Pipeline (FTG Style Braking)
        angular_vel = max(-self.max_angular_vel, min(self.max_angular_vel, angular_vel))
        
        # Reduce speed proportionally to steering effort
        turn_ratio = abs(angular_vel) / self.max_angular_vel
        speed = self.max_velocity * (1.0 - 0.65 * turn_ratio)

        # Emergency frontal braking
        if min_front_dist <= 0.35:
            speed = 0.0
        elif min_front_dist < 1.0:
            speed *= max(0.0, min(1.0, (min_front_dist - 0.35) / (1.0 - 0.35)))

        self.cmd_pub.publish(self.make_command(max(0.0, speed), angular_vel))

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
    node = OptimizedWallFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()