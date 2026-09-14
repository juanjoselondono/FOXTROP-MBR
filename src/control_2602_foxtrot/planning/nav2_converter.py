#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped

class CmdVelConverter(Node):
    def __init__(self):
        super().__init__('nav2_cmd_vel_converter')
        # Nav2 final output is always /cmd_vel (as Twist)
        self.sub = self.create_subscription(Twist, '/cmd_vel', self.listener_callback, 10)
        # Gazebo diff_drive_controller expects /diffdrive_controller/cmd_vel (as TwistStamped)
        self.pub = self.create_publisher(TwistStamped, '/diffdrive_controller/cmd_vel', 10)
        self.get_logger().info("Nav2 Twist to TwistStamped Converter Started!")

    def listener_callback(self, msg):
        stamped_msg = TwistStamped()
        stamped_msg.header.stamp = self.get_clock().now().to_msg()
        stamped_msg.header.frame_id = 'base_link'
        stamped_msg.twist = msg
        self.pub.publish(stamped_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

