import re

with open('src/control_2602_foxtrot/planning/astar_planner.py', 'r') as f:
    code = f.read()

# 1. Add race_mode and waypoints to __init__
init_hook = "self.declare_parameter('goal_y', -0.6)"
init_addition = """
        self.race_mode = False
        self.race_waypoints = [
            (-15.3759, 4.45454),
            (-0.441745, -0.787758)
        ]
        self.current_wp_idx = 0
"""
code = code.replace(init_hook, init_hook + "\n" + init_addition)

# 2. Replace plan_timer_callback
timer_code = """    def plan_timer_callback(self):
        if not self.race_mode:
            return
            
        rx, ry, _ = self.get_robot_pose()
        if rx is None or self.current_goal is None:
            return
            
        # Check distance to current goal
        goal_x = self.current_goal.pose.position.x
        goal_y = self.current_goal.pose.position.y
        dist = math.hypot(goal_x - rx, goal_y - ry)
        
        # If we are within 3.5m of the waypoint, seamlessly calculate the path to the next one
        if dist < 3.5:
            self.advance_waypoint()"""
code = re.sub(r'    def plan_timer_callback\(self\):\n\s+pass.*?\n', timer_code + '\n', code)

# 3. Add advance_waypoint and replace laps_callback
laps_code = """    def advance_waypoint(self):
        rx, ry, _ = self.get_robot_pose()
        if rx is None: return
        
        goal_x, goal_y = self.race_waypoints[self.current_wp_idx]
        
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = float(goal_x)
        goal.pose.position.y = float(goal_y)
        goal.pose.orientation.w = 1.0
        self.current_goal = goal
        
        self.get_logger().info(f"F1 Mode: En route to Waypoint {self.current_wp_idx + 1}/2 ({goal_x:.2f}, {goal_y:.2f}). Laps left: {self.laps_remaining}")
        
        # Advance index for NEXT time
        self.current_wp_idx += 1
        if self.current_wp_idx >= len(self.race_waypoints):
            self.current_wp_idx = 0
            self.laps_remaining -= 1
            if self.laps_remaining < 0:
                self.race_mode = False
                self.get_logger().info("Race Finished!")
                return
                
        self.plan_path()

    def laps_callback(self, msg):
        self.laps_remaining = msg.data
        if self.laps_remaining <= 0:
            self.race_mode = False
            return
            
        self.race_mode = True
        self.current_wp_idx = 0
        self.advance_waypoint()"""

# We need to replace laps_callback and lap_completed_callback with the new logic
# We can just replace from `def lap_completed_callback` to `def get_robot_pose`
pattern = r'    def lap_completed_callback\(self, msg\):.*?def get_robot_pose\(self\):'
code = re.sub(pattern, laps_code + '\n\n    def get_robot_pose(self):', code, flags=re.DOTALL)

# 4. In plan_path, set dynamic_wall if race_mode
plan_path_mod = """        # Use existing dynamic_wall if set by race mode, otherwise leave it None
        if getattr(self, 'race_mode', False):
            self.dynamic_wall = (rx, ry, ryaw)"""
code = code.replace("        # Use existing dynamic_wall if set by race mode, otherwise leave it None", plan_path_mod)

with open('src/control_2602_foxtrot/planning/astar_planner.py', 'w') as f:
    f.write(code)

