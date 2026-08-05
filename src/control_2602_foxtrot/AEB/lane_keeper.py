#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneKeeperNode(Node):
    def __init__(self):
        super().__init__('lane_keeper_node')
        
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.state_sub = self.create_subscription(
            Bool, '/lk/state', self.state_callback, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel_nav', 10)
        
        self.lk_enabled = False
        
        # PD Control Gains
        self.kp = 0.003
        self.kd = 0.0015
        self.forward_velocity = 0.6  # Increased speed for smooth tracking
        
        self.prev_error = 0.0
        self.last_time = 0.0
        
        # Memory buffer for missing lines
        self.assumed_lane_width = 300 # Estimated width of the lane in warped pixels
        
        self.get_logger().info("Advanced Dual-Boundary Lane Keeper Online.")

    def state_callback(self, msg):
        if self.lk_enabled and not msg.data:
            self.stop_vehicle()
        self.lk_enabled = msg.data

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        height, width = frame.shape[:2]
        
        # ==========================================
        # 1. Perspective Transform (Corrected ROI)
        # ==========================================
        # Lifted the top points higher to see down the road, widened the base
        src_points = np.float32([
            [width * 0.1, height * 0.55],  # Top Left
            [width * 0.9, height * 0.55],  # Top Right
            [width * 1.0, height * 0.95],  # Bottom Right
            [0.0,         height * 0.95]   # Bottom Left
        ])
        
        dst_points = np.float32([
            [0, 0], [width, 0], [width, height], [0, height]
        ])
        
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(frame, matrix, (width, height))

        # ==========================================
        # 2. Dual-Color Thresholding
        # ==========================================
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        
        # Isolate Yellow (Left boundary)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Isolate White (Right shoulder)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Split screens: Yellow only valid on left, White only valid on right
        left_mask = mask_yellow.copy()
        left_mask[:, width//2:] = 0  
        
        right_mask = mask_white.copy()
        right_mask[:, :width//2] = 0 

        # ==========================================
        # 3. Smart Centroid Logic
        # ==========================================
        M_left = cv2.moments(left_mask)
        M_right = cv2.moments(right_mask)
        
        cx_left = int(M_left["m10"] / M_left["m00"]) if M_left["m00"] > 0 else None
        cx_right = int(M_right["m10"] / M_right["m00"]) if M_right["m00"] > 0 else None
        
        target_center = None

        # Logic A: Both lines visible (Calibrate lane width)
        if cx_left is not None and cx_right is not None:
            target_center = (cx_left + cx_right) // 2
            self.assumed_lane_width = cx_right - cx_left # Dynamically update memory
            cv2.circle(warped, (cx_left, height // 2), 8, (0, 255, 255), -1)
            cv2.circle(warped, (cx_right, height // 2), 8, (255, 255, 255), -1)

        # Logic B: Only White shoulder visible (Yellow dashed line missing)
        elif cx_right is not None:
            target_center = cx_right - (self.assumed_lane_width // 2)
            cv2.circle(warped, (cx_right, height // 2), 8, (255, 255, 255), -1)

        # Logic C: Only Yellow line visible
        elif cx_left is not None:
            target_center = cx_left + (self.assumed_lane_width // 2)
            cv2.circle(warped, (cx_left, height // 2), 8, (0, 255, 255), -1)

        # ==========================================
        # 4. Error Calculus & Execution
        # ==========================================
        if target_center is not None:
            # Draw the calculated target trajectory in green
            cv2.circle(warped, (target_center, height // 2), 10, (0, 255, 0), -1)
            
            # The mathematical center of the camera is our chassis heading
            camera_center = width // 2
            error = camera_center - target_center
            
            if self.lk_enabled:
                self.execute_control(error)
        else:
            if self.lk_enabled:
                self.get_logger().warn("Complete Track Loss. Halting.", throttle_duration_sec=1.0)
                self.stop_vehicle()

        # Debug Visualization
        pts = src_points.astype(int).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Combine masks for the debug window to see both detections
        combined_mask = cv2.bitwise_or(left_mask, right_mask)
        
        cv2.imshow("Raw Optics", frame)
        cv2.imshow("Dual Boundary Bird's Eye", warped)
        cv2.imshow("Isolations (Yellow Left, White Right)", combined_mask)
        cv2.waitKey(1)

    def execute_control(self, error):
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_time == 0.0:
            self.last_time = current_time
            return
            
        dt = current_time - self.last_time
        if dt <= 0.0:
            return
            
        error_dot = (error - self.prev_error) / dt
        steering = (self.kp * error) + (self.kd * error_dot)
        
        self.prev_error = error
        self.last_time = current_time
        
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = self.forward_velocity
        cmd.twist.angular.z = float(steering)
        self.cmd_pub.publish(cmd)

    def stop_vehicle(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = 0.0
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = LaneKeeperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()