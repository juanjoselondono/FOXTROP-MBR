#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

class ModeSupervisor(Node):
    def __init__(self):
        super().__init__('mode_supervisor')
        
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        self.aeb_pub = self.create_publisher(Bool, '/aeb/state', 10)
        self.wf_pub = self.create_publisher(Bool, '/wf/state', 10)

        self.aeb_active = True
        self.wf_active = False
        self.last_buttons = None

        # A 1Hz heartbeat timer to ensure nodes sync without flooding the network
        self.sync_timer = self.create_timer(1.0, self.publish_states)

        self.get_logger().info("Event-Driven Supervisor Online. A = Auto, X = Manual, B = Toggle AEB")

    def joy_callback(self, msg):
        if self.last_buttons is None:
            self.last_buttons = msg.buttons
            return

        state_changed = False

        # BUTTON X (Index 0): Enable Manual Navigation
        if msg.buttons[0] == 1 and self.last_buttons[0] == 0:
            self.wf_active = False
            self.get_logger().info("MODE ENGAGED: Manual Joystick Navigation.")
            state_changed = True

        # BUTTON A (Index 1): Enable Autonomous Wall Following
        if msg.buttons[1] == 1 and self.last_buttons[1] == 0:
            self.wf_active = True
            self.get_logger().info("MODE ENGAGED: Autonomous Wall Following.")
            state_changed = True

        # BUTTON B (Index 2): Toggle AEB System On/Off
        if msg.buttons[2] == 1 and self.last_buttons[2] == 0:
            self.aeb_active = not self.aeb_active
            status = "ENABLED" if self.aeb_active else "DISABLED"
            self.get_logger().warn(f"AEB System {status}.")
            state_changed = True

        # ONLY publish immediately if a physical button was triggered
        if state_changed:
            self.publish_states()

        self.last_buttons = msg.buttons

    def publish_states(self):
        aeb_msg = Bool()
        aeb_msg.data = self.aeb_active
        self.aeb_pub.publish(aeb_msg)

        wf_msg = Bool()
        wf_msg.data = self.wf_active
        self.wf_pub.publish(wf_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ModeSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()