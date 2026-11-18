"""Shared local trajectory-context features for training and inference."""
from __future__ import annotations

import numpy as np

from pathsnap import ProjectedGraph, Trip


CONTEXT_FEATURE_NAMES = [
    'smoothed_heading_error_rad',
    'previous_point_to_arc_distance_m',
    'next_point_to_arc_distance_m',
    'neighbor_mean_point_to_arc_distance_m',
    'point_two_samples_back_to_arc_distance_m',
    'point_two_samples_ahead_to_arc_distance_m',
    'five_point_window_minimum_distance_m',
    'five_point_window_mean_distance_m',
]


def add_local_features(trip: Trip, x: np.ndarray, arc_ids: np.ndarray,
                       obs_ids: np.ndarray, context_radius: int = 1) -> np.ndarray:
    if context_radius not in (1, 2):
        raise ValueError('context_radius must be 1 or 2')
    graph = ProjectedGraph(trip)
    extras = np.empty((len(obs_ids), 4 if context_radius == 1 else 8), dtype=np.float32)
    row_batch = 500_000
    last_point = len(graph.track_xy) - 1
    for start in range(0, len(obs_ids), row_batch):
        end = min(len(obs_ids), start + row_batch)
        point_ids = obs_ids[start:end]
        edge_ids = arc_ids[start:end]
        edge_vec = graph.vec[edge_ids]
        edge_start = graph.start[edge_ids]
        denominator = np.sum(edge_vec * edge_vec, axis=1).clip(1e-8)

        def offset_distance(offset: int) -> np.ndarray:
            neighbor_points = graph.track_xy[np.clip(point_ids + offset, 0, last_point)]
            delta = neighbor_points - edge_start
            fraction = np.clip(np.sum(delta * edge_vec, axis=1) / denominator, 0.0, 1.0)
            projection = edge_start + fraction[:, None] * edge_vec
            return np.linalg.norm(projection - neighbor_points, axis=1)

        previous_distance = offset_distance(-1)
        next_distance = offset_distance(1)
        wide_previous = graph.track_xy[np.maximum(0, point_ids - 2)]
        wide_next = graph.track_xy[np.minimum(last_point, point_ids + 2)]
        direction = wide_next - wide_previous
        moving = np.linalg.norm(direction, axis=1) >= 1.0
        heading_error = np.full(len(point_ids), np.pi / 2, dtype=np.float64)
        heading = np.arctan2(direction[moving, 1], direction[moving, 0])
        moving_angle = np.abs((graph.heading[edge_ids[moving]] - heading + np.pi) % (2 * np.pi) - np.pi)
        heading_error[moving] = moving_angle
        extras[start:end, 0] = heading_error
        extras[start:end, 1] = previous_distance
        extras[start:end, 2] = next_distance
        extras[start:end, 3] = (previous_distance + next_distance) / 2.0
        if context_radius == 2:
            previous_two = offset_distance(-2)
            next_two = offset_distance(2)
            window = np.column_stack((previous_two, previous_distance, x[start:end, 0],
                                      next_distance, next_two))
            extras[start:end, 4] = previous_two
            extras[start:end, 5] = next_two
            extras[start:end, 6] = np.min(window, axis=1)
            extras[start:end, 7] = np.mean(window, axis=1)
    return np.column_stack((x.astype(np.float32), extras))
