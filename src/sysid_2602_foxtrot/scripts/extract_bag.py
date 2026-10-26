#!/usr/bin/env python3
"""
extract_bag.py
==============
Read a rosbag (.mcap) and export synchronized data to CSV files.

Usage:
    python3 extract_bag.py <path_to_bag_folder>

Output files (written to the current working directory):
    pose_raw.csv          — VICON pose  [t, x, y, z, yaw]
    esc_command_raw.csv   — ESC command [t, u]
    joy_raw.csv           — Joystick    [t, joy_throttle, joy_steering]

Dependencies:
    pip install rosbags numpy pandas
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

BAG_PATH   = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
TYPESTORE  = get_typestore(Stores.ROS2_JAZZY)

# ── Topic names — adjust to your actual bag if needed ──────────────────────────
TOPIC_POSE = '/robot1/pose'    # geometry_msgs/PoseStamped from VICON bridge
TOPIC_ESC  = '/esc/command'    # std_msgs/Float32  normalised throttle command
TOPIC_JOY  = '/joy'            # sensor_msgs/Joy

# ─────────────────────────────────────────────────────────────────────────────

def read_topic(reader, typestore, topic_name, fields_fn):
    """Extract (timestamp_s, *fields) rows from a single topic."""
    rows = []
    connections = [c for c in reader.connections if c.topic == topic_name]
    if not connections:
        print(f'  WARNING: topic "{topic_name}" not found in bag')
        return pd.DataFrame()
    for conn, timestamp, rawdata in reader.messages(connections=connections):
        msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
        t   = timestamp * 1e-9          # nanoseconds → seconds
        rows.append((t,) + fields_fn(msg))
    return pd.DataFrame(rows)


def extract_pose(msg):
    p = msg.pose.position
    o = msg.pose.orientation
    yaw = np.arctan2(
        2.0 * (o.w * o.z + o.x * o.y),
        1.0 - 2.0 * (o.y * o.y + o.z * o.z)
    )
    return (float(p.x), float(p.y), float(p.z), float(yaw))


def extract_esc_command(msg):
    return (float(msg.data),)


def extract_joy(msg):
    # axes[1] = left stick vertical (throttle), axes[3] = right stick horizontal (steering)
    throttle = float(msg.axes[1]) if len(msg.axes) > 1 else 0.0
    steering = float(msg.axes[3]) if len(msg.axes) > 3 else 0.0
    return (throttle, steering)


# ── Main ─────────────────────────────────────────────────────────────────────

with Reader(BAG_PATH) as reader:
    print(f'Reading bag: {BAG_PATH}')
    print('Topics found in bag:')
    for c in reader.connections:
        print(f'  {c.topic}  ({c.msgtype})')

    df_pose = read_topic(reader, TYPESTORE, TOPIC_POSE, extract_pose)
    df_cmd  = read_topic(reader, TYPESTORE, TOPIC_ESC,  extract_esc_command)
    df_joy  = read_topic(reader, TYPESTORE, TOPIC_JOY,  extract_joy)

# Assign column names (guard for empty DataFrames)
if not df_pose.empty:
    df_pose.columns = ['t', 'x', 'y', 'z', 'yaw']
if not df_cmd.empty:
    df_cmd.columns  = ['t', 'u']
if not df_joy.empty:
    df_joy.columns  = ['t', 'joy_throttle', 'joy_steering']

# Sort by time
for df in [df_pose, df_cmd, df_joy]:
    if not df.empty:
        df.sort_values('t', inplace=True)
        df.reset_index(drop=True, inplace=True)

# Align time to t=0
valid_starts = [df['t'].iloc[0] for df in [df_pose, df_cmd, df_joy] if not df.empty]
if valid_starts:
    t0 = min(valid_starts)
    for df in [df_pose, df_cmd, df_joy]:
        if not df.empty:
            df['t'] -= t0

# Save raw CSVs
df_pose.to_csv('pose_raw.csv', index=False)
df_cmd.to_csv('esc_command_raw.csv', index=False)
df_joy.to_csv('joy_raw.csv', index=False)

# Stats
def stats(df, name):
    if df.empty:
        print(f'  {name}: (no data)')
    else:
        dt_ms = df['t'].diff().median() * 1000
        print(f'  {name}: {len(df)} samples, median dt={dt_ms:.1f} ms')

print('\nExtraction complete:')
stats(df_pose, 'VICON pose ')
stats(df_cmd,  'ESC command')
stats(df_joy,  'Joystick   ')
print('Saved: pose_raw.csv, esc_command_raw.csv, joy_raw.csv')
