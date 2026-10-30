#!/usr/bin/env python3
"""
line_extraction_node.py
=======================
Line segment extraction from a 2D LiDAR scan (split-and-merge / IEPF):

  1. polar -> Cartesian in the scan frame, invalid/out-of-range beams dropped
  2. break into clusters wherever consecutive points are > cluster_gap apart
  3. split each cluster recursively at the point farthest from the chord,
     while that distance exceeds split_distance
  4. fit every piece by total least squares (PCA) and keep it if it has
     >= min_points and is >= min_length long
  5. merge neighbouring segments that are collinear (angle < merge_angle and
     each endpoint within merge_distance of the other line)

Publishes /line_segments (visualization_msgs/MarkerArray, LINE_LIST in the
scan frame): one marker holding every segment as consecutive point pairs.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


def scan_to_points(ranges, angle_min, angle_inc, r_min, r_max):
    r = np.asarray(ranges, dtype=float)
    a = angle_min + angle_inc * np.arange(len(r))
    ok = np.isfinite(r) & (r >= r_min) & (r <= r_max)
    return np.column_stack((r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])))


def cluster(points, gap, wraps=False):
    """wraps: the scan covers a full turn, so the last and first beams are
    neighbours -- a wall crossing the scan seam must not be cut in two."""
    if len(points) == 0:
        return []
    jumps = np.flatnonzero(np.linalg.norm(np.diff(points, axis=0), axis=1) > gap) + 1
    clusters = np.split(points, jumps)
    if wraps and len(clusters) > 1 and np.linalg.norm(points[-1] - points[0]) <= gap:
        clusters[0] = np.vstack((clusters.pop(), clusters[0]))
    return clusters


def split(points, threshold, min_points):
    """Iterative split: returns index ranges (start, end inclusive) of pieces."""
    out, stack = [], [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i + 1 < min_points:
            continue
        p0, p1 = points[i], points[j]
        d = p1 - p0
        n = np.hypot(d[0], d[1])
        seg = points[i:j + 1]
        if n < 1e-9:
            dist = np.linalg.norm(seg - p0, axis=1)
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p0[1]) - d[1] * (seg[:, 0] - p0[0])) / n
        k = int(np.argmax(dist))
        if dist[k] > threshold and 0 < k < j - i:
            stack.append((i, i + k))
            stack.append((i + k, j))
        else:
            out.append((i, j))
    return sorted(out)


def fit_segment(points):
    """Total-least-squares line through points; endpoints = extreme projections."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    u = vt[0]
    t = (points - c) @ u
    return c + t.min() * u, c + t.max() * u


def _angle(s):
    d = s[1] - s[0]
    return math.atan2(d[1], d[0])


def _point_line_dist(p, s):
    d = s[1] - s[0]
    n = math.hypot(d[0], d[1])
    return abs(d[0] * (p[1] - s[0][1]) - d[1] * (p[0] - s[0][0])) / n if n > 1e-9 else math.dist(p, s[0])


def _mergeable(m, s, max_angle, max_dist):
    dang = abs((_angle(s) - _angle(m) + math.pi / 2) % math.pi - math.pi / 2)
    return (dang < max_angle and _point_line_dist(s[0], m) < max_dist and _point_line_dist(s[1], m) < max_dist
            and math.dist(m[1], s[0]) < 4 * max_dist + 0.3)


def merge(segments, max_angle, max_dist, wraps=False):
    if not segments:
        return []
    merged = [segments[0]]
    for s in segments[1:]:
        if _mergeable(merged[-1], s, max_angle, max_dist):
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    # full-turn scan: the wall under the scan seam shows up as last + first segment
    if wraps and len(merged) > 1 and _mergeable(merged[-1], merged[0], max_angle, max_dist):
        last = merged.pop()
        merged[0] = (last[0], merged[0][1])
    return merged


def extract_lines(points, gap, split_distance, min_points, min_length, merge_angle, merge_distance,
                  wraps=False):
    segments = []
    for cl in cluster(points, gap, wraps):
        for i, j in split(cl, split_distance, min_points):
            a, b = fit_segment(cl[i:j + 1])
            if math.dist(a, b) >= min_length:
                segments.append((a, b))
    return merge(segments, merge_angle, merge_distance, wraps)


class LineExtractionNode(Node):

    def __init__(self):
        super().__init__('line_extraction_node')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('output_topic', '/line_segments')
        self.declare_parameter('min_range', 0.05)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('cluster_gap', 0.15)        # [m] break between clusters
        self.declare_parameter('split_distance', 0.03)     # [m] max point-to-chord deviation
        self.declare_parameter('min_points', 6)
        self.declare_parameter('min_length', 0.15)         # [m]
        self.declare_parameter('merge_angle_deg', 5.0)
        self.declare_parameter('merge_distance', 0.05)     # [m]

        p = lambda n: self.get_parameter(n).value
        self._cfg = dict(gap=float(p('cluster_gap')), split_distance=float(p('split_distance')),
                         min_points=int(p('min_points')), min_length=float(p('min_length')),
                         merge_angle=math.radians(float(p('merge_angle_deg'))),
                         merge_distance=float(p('merge_distance')))
        self._r_min, self._r_max = float(p('min_range')), float(p('max_range'))
        self._pub = self.create_publisher(MarkerArray, p('output_topic'), 10)
        self.create_subscription(LaserScan, p('scan_topic'), self._scan_cb, qos_profile_sensor_data)

    def _scan_cb(self, scan: LaserScan):
        pts = scan_to_points(scan.ranges, scan.angle_min, scan.angle_increment,
                             max(self._r_min, scan.range_min), min(self._r_max, scan.range_max))
        span = scan.angle_increment * len(scan.ranges)
        segments = extract_lines(pts, **self._cfg, wraps=span >= 2.0 * math.pi - 2.0 * scan.angle_increment)

        m = Marker()
        m.header = scan.header
        m.ns = 'line_segments'
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.03
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 1.0
        for a, b in segments:
            m.points.append(Point(x=float(a[0]), y=float(a[1]), z=0.0))
            m.points.append(Point(x=float(b[0]), y=float(b[1]), z=0.0))
        self._pub.publish(MarkerArray(markers=[m]))


def main(args=None):
    rclpy.init(args=args)
    node = LineExtractionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
