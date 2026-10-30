#!/usr/bin/env python3
"""Export the EKF / laser-odometry / SLAM topics of a slam_ekf bag to CSV and plot them.

Usage:
    python3 bag_to_csv.py slam_bags/slam_ekf_20261029_180955 [--map maps/lab.yaml]

Writes <bag>/csv/{ekf,rf2o,slam_pose,imu,cmd}.csv and <bag>/csv/ekf_vs_slam.png.

Frames: /odometry/filtered and /odom_rf2o are in `odom`; /pose (slam_toolbox) is in `map`.
The EKF trajectory is drawn in `map` by composing it with the map->odom TF from /tf.
"""
import argparse
import bisect
import csv
import math
import os

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def read_bag(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=''),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        yield topic, deserialize_message(data, types[topic])


def odom_row(m):
    p, t = m.pose.pose, m.twist.twist
    c = m.pose.covariance
    return [stamp(m.header), p.position.x, p.position.y, yaw_of(p.orientation),
            t.linear.x, t.linear.y, t.angular.z, c[0], c[7], c[35]]


ODOM_COLS = ['t', 'x', 'y', 'yaw', 'vx', 'vy', 'wz', 'var_x', 'var_y', 'var_yaw']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--map', default=None, help='map .yaml to draw under the trajectories')
    a = ap.parse_args()

    rows = {'ekf': [], 'rf2o': [], 'slam_pose': [], 'imu': [], 'cmd': []}
    map_odom = []   # (t, x, y, yaw) of map->odom
    for topic, m in read_bag(a.bag):
        if topic == '/odometry/filtered':
            rows['ekf'].append(odom_row(m))
        elif topic == '/odom_rf2o':
            rows['rf2o'].append(odom_row(m))
        elif topic == '/pose':
            p = m.pose.pose
            c = m.pose.covariance
            rows['slam_pose'].append([stamp(m.header), p.position.x, p.position.y, yaw_of(p.orientation),
                                      c[0], c[7], c[35]])
        elif topic == '/imu/data_raw':
            rows['imu'].append([stamp(m.header), m.angular_velocity.z,
                                m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        elif topic == '/cmd_vel_stamped':
            rows['cmd'].append([stamp(m.header), m.twist.linear.x, m.twist.angular.z])
        elif topic == '/tf':
            for tr in m.transforms:
                if tr.header.frame_id == 'map' and tr.child_frame_id == 'odom':
                    q, v = tr.transform.rotation, tr.transform.translation
                    map_odom.append((stamp(tr.header), v.x, v.y, yaw_of(q)))

    out = os.path.join(a.bag, 'csv')
    os.makedirs(out, exist_ok=True)
    cols = {'ekf': ODOM_COLS, 'rf2o': ODOM_COLS,
            'slam_pose': ['t', 'x', 'y', 'yaw', 'var_x', 'var_y', 'var_yaw'],
            'imu': ['t', 'wz', 'ax', 'ay', 'az'], 'cmd': ['t', 'linear_x', 'angular_z']}
    for name, r in rows.items():
        r.sort(key=lambda row: row[0])
        with open(os.path.join(out, f'{name}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(cols[name])
            w.writerows(r)
        print(f'{name:10s} {len(r):5d} rows -> {out}/{name}.csv')

    # ── EKF expressed in map, using the latest map->odom at or before each sample ──
    map_odom.sort()
    mo_t = [s[0] for s in map_odom]
    ekf_map = []
    for t, x, y, yaw, *_ in rows['ekf']:
        i = bisect.bisect_right(mo_t, t) - 1
        if i < 0:
            continue
        _, ox, oy, oyaw = map_odom[i]
        c, s = math.cos(oyaw), math.sin(oyaw)
        ekf_map.append((t, ox + c * x - s * y, oy + s * x + c * y))
    ekf_map = np.array(ekf_map)

    # ── EKF (in map) vs SLAM pose at the SLAM timestamps ──
    sp = np.array(rows['slam_pose'])
    err = []
    if len(ekf_map) and len(sp):
        for t, x, y, *_ in sp:
            if ekf_map[0, 0] <= t <= ekf_map[-1, 0]:
                ex = np.interp(t, ekf_map[:, 0], ekf_map[:, 1])
                ey = np.interp(t, ekf_map[:, 0], ekf_map[:, 2])
                err.append(math.hypot(ex - x, ey - y))
    ekf = np.array(rows['ekf'])
    dist = float(np.sum(np.hypot(np.diff(ekf[:, 1]), np.diff(ekf[:, 2])))) if len(ekf) > 1 else 0.0
    print(f'\nEKF path length (odom): {dist:.2f} m')
    if map_odom:
        _, ox, oy, oyaw = map_odom[-1]
        print(f'final map->odom correction: {math.hypot(ox, oy):.3f} m, {math.degrees(oyaw):.1f} deg '
              f'({100 * math.hypot(ox, oy) / max(dist, 1e-6):.1f} % of distance)')
    if err:
        print(f'EKF(map) vs SLAM /pose: mean {np.mean(err):.3f} m, max {np.max(err):.3f} m, n={len(err)}')

    # ── plot ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    if a.map:
        meta = yaml.safe_load(open(a.map))
        img = plt.imread(os.path.join(os.path.dirname(a.map), meta['image']))
        res, (x0, y0, _) = meta['resolution'], meta['origin']
        h, w = img.shape[:2]
        ax[0].imshow(img, cmap='gray', origin='upper', extent=[x0, x0 + w * res, y0, y0 + h * res])
    rf = np.array(rows['rf2o'])
    if len(rf):
        ax[0].plot(rf[:, 1], rf[:, 2], ':', label='rf2o (odom)')
    if len(ekf):
        ax[0].plot(ekf[:, 1], ekf[:, 2], '--', label='EKF (odom)')
    if len(ekf_map):
        ax[0].plot(ekf_map[:, 1], ekf_map[:, 2], label='EKF (map)')
    if len(sp):
        ax[0].plot(sp[:, 1], sp[:, 2], 'o', ms=3, label='SLAM /pose (map)')
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('x [m]')
    ax[0].set_ylabel('y [m]')
    ax[0].legend()
    ax[0].set_title('Trajectories')
    if len(ekf):
        t0 = ekf[0, 0]
        ax[1].plot(ekf[:, 0] - t0, ekf[:, 4], label='EKF vx [m/s]')
        ax[1].plot(ekf[:, 0] - t0, ekf[:, 6], label='EKF wz [rad/s]')
        if len(rf):
            ax[1].plot(rf[:, 0] - t0, rf[:, 4], ':', label='rf2o vx [m/s]')
        ax[1].set_xlabel('t [s]')
        ax[1].legend()
        ax[1].set_title('EKF velocities')
        ax[1].grid(True)
    fig.tight_layout()
    png = os.path.join(out, 'ekf_vs_slam.png')
    fig.savefig(png, dpi=120)
    print(f'plot -> {png}')


if __name__ == '__main__':
    main()
