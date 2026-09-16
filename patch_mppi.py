import re

with open('src/control_2602_foxtrot/tracking/mppi_tracker.py', 'r') as f:
    code = f.read()

# Replace the Laps Logic stop block
old_stop = """        # Only stop if we are on the FINAL lap!
        if self.laps_remaining <= 0:
            if dist_to_goal < 0.35 or passed_goal:
                self.stop_vehicle()
                if passed_goal:
                    self.get_logger().info("Meta superada. Anclaje de emergencia activado.")
                else:
                    self.get_logger().info("Meta alcanzada. Freno total activado.")
                return"""

new_stop = """        # If we reach the end of the currently active path, STOP.
        # (During a race, A* updates the path 3.5m before we reach the end, so we never stop until the very end)
        if dist_to_goal < 0.35 or passed_goal:
            self.stop_vehicle()
            if passed_goal:
                self.get_logger().info("Meta superada. Anclaje de emergencia activado.")
            else:
                self.get_logger().info("Meta alcanzada. Freno total activado.")
            return"""
            
code = code.replace(old_stop, new_stop)

with open('src/control_2602_foxtrot/tracking/mppi_tracker.py', 'w') as f:
    f.write(code)

