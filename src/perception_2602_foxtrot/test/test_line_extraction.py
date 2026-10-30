"""Tests for the split-and-merge line extractor on synthetic scans."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from perception_2602_foxtrot.line_extraction_node import extract_lines, scan_to_points  # noqa: E402

CFG = dict(gap=0.15, split_distance=0.03, min_points=6, min_length=0.15,
           merge_angle=math.radians(5.0), merge_distance=0.05)


def ray_scan(walls, n=720, noise=0.0, seed=0):
    """Ranges of a 360 deg scan inside a set of infinite-plane walls ((nx, ny), d)."""
    rng = np.random.default_rng(seed)
    a = -math.pi + 2 * math.pi * np.arange(n) / n
    r = np.full(n, np.inf)
    for (nx, ny), d in walls:
        denom = np.cos(a) * nx + np.sin(a) * ny
        with np.errstate(divide='ignore'):
            t = d / denom
        r = np.where((denom > 1e-9) & (t < r), t, r)
    return r + noise * rng.standard_normal(n), -math.pi, 2 * math.pi / n


def test_room_gives_four_walls():
    r, a0, da = ray_scan([((1, 0), 2.0), ((-1, 0), 1.5), ((0, 1), 1.0), ((0, -1), 1.2)], noise=0.005)
    segs = extract_lines(scan_to_points(r, a0, da, 0.05, 8.0), **CFG, wraps=True)
    assert len(segs) == 4
    lengths = sorted(math.dist(a, b) for a, b in segs)
    assert lengths[-1] > 3.0      # the 3.5 m long walls (x from -1.5 to 2.0)


def test_single_wall_geometry():
    r, a0, da = ray_scan([((1, 0), 2.0)], noise=0.003)
    pts = scan_to_points(r, a0, da, 0.05, 5.0)
    segs = extract_lines(pts, **CFG)
    assert len(segs) == 1
    (x0, _), (x1, _) = segs[0]
    assert abs(x0 - 2.0) < 0.02 and abs(x1 - 2.0) < 0.02


def test_empty_scan():
    assert extract_lines(np.zeros((0, 2)), **CFG) == []
