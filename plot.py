import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
import math
import sys

def yaw_from_quaternion(q):
    """Extracts the yaw (Z) angle from a ROS 2 geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def normalize_trajectory(traj_array):
    """Translates and rotates a trajectory to start exactly at (0,0) facing 0 radians."""
    # traj_array format: [timestamp, x, y, yaw]
    x0, y0, yaw0 = traj_array[0, 1], traj_array[0, 2], traj_array[0, 3]
    
    # Translate to origin
    dx = traj_array[:, 1] - x0
    dy = traj_array[:, 2] - y0
    
    # Rotate by -yaw0 to align heading
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    x_norm = dx * c - dy * s
    y_norm = dx * s + dy * c
    
    return x_norm, y_norm

# =========================================================================
# 1. Configuration
# =========================================================================
bag_path = 'odom_ekf_comparison' 
ekf_topic = '/ekf/odometry'
odom_topic = '/diffdrive_controller/odom'

ekf_raw, odom_raw = [], []

print(f"Parsing offline MCAP file in: {bag_path} ...")

try:
    with AnyReader([Path(bag_path)]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == ekf_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                yaw = yaw_from_quaternion(msg.pose.pose.orientation)
                # Store timestamp alongside spatial data
                ekf_raw.append([timestamp, msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
            elif connection.topic == odom_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                yaw = yaw_from_quaternion(msg.pose.pose.orientation)
                odom_raw.append([timestamp, msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
except Exception as e:
    print(f"System error reading bag: {e}")
    sys.exit(1)

if not ekf_raw or not odom_raw:
    print(f"Error: Could not find data on {ekf_topic} or {odom_topic}.")
    sys.exit(1)

ekf_raw = np.array(ekf_raw)
odom_raw = np.array(odom_raw)

# =========================================================================
# 2. Time Synchronization
# =========================================================================
print("Synchronizing telemetry timestamps...")
# Find the latest start time so both datasets begin exactly together
start_time = max(ekf_raw[0, 0], odom_raw[0, 0])

# Truncate any asynchronous lead-in data
ekf_sync = ekf_raw[ekf_raw[:, 0] >= start_time]
odom_sync = odom_raw[odom_raw[:, 0] >= start_time]

if len(ekf_sync) == 0 or len(odom_sync) == 0:
    print("Error: No overlapping timestamps found between topics.")
    sys.exit(1)

# =========================================================================
# 3. Normalize Trajectories
# =========================================================================
print("Aligning coordinate frames...")
ekf_x, ekf_y = normalize_trajectory(ekf_sync)
odom_x, odom_y = normalize_trajectory(odom_sync)

# =========================================================================
# 4. Dynamic Metric Calculations
# =========================================================================
dx_diff, dy_diff = np.diff(ekf_x), np.diff(ekf_y)
dist_driven = np.sum(np.sqrt(dx_diff**2 + dy_diff**2))

final_drift = math.hypot(odom_x[-1] - ekf_x[-1], odom_y[-1] - ekf_y[-1])

print(f"Alignment complete. True Drift: {final_drift:.2f}m over {dist_driven:.1f}m.")

# =========================================================================
# 5. Trajectory Visualization
# =========================================================================
plt.figure(figsize=(8, 5.5))

plt.plot(ekf_x, ekf_y, '-', color='#2B82BA', linewidth=2.5, label='EKF Odometry')
plt.plot(odom_x, odom_y, '--', color='#F28C2B', linewidth=2.5, label='/odom (dead reckoning)')

plt.plot(0, 0, 'ko', markersize=7, label='Start')

plt.title(f'Dead-reckoning drift vs EKF: {final_drift:.2f} m after {dist_driven:.1f} m driven', fontsize=14)
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
plt.axis('equal')
plt.grid(True, alpha=0.4)
plt.legend(loc='lower right', fontsize=11)

plt.tight_layout()
plt.show()