"""PathSnap map matching: spatial candidate scoring plus graph-aware Viterbi."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

EARTH_M = 6_371_000.0


def _read_numeric(path: Path) -> np.ndarray:
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric data in {path}") from exc
    if values.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    return np.atleast_2d(values)


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray | None = None

    @classmethod
    def from_directory(cls, directory: str | Path, *, include_route: bool = True) -> "Trip":
        folder = Path(directory)
        trip_id = folder.name
        required = [folder / f"{trip_id}.{ext}" for ext in ("track", "nodes", "arcs")]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing trip file(s): " + ", ".join(missing))
        route_file = folder / f"{trip_id}.route"
        route = (_read_numeric(route_file).reshape(-1).astype(np.int64)
                 if include_route and route_file.exists() else None)
        trip = cls(trip_id, _read_numeric(required[0]), _read_numeric(required[1]),
                   _read_numeric(required[2]), route)
        trip.validate()
        return trip

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Trip":
        def arr(key: str, cols: int | None = None) -> np.ndarray:
            value = np.asarray(payload[key], dtype=np.float64)
            value = np.atleast_2d(value)
            if cols is not None and value.shape[1] < cols:
                raise ValueError(f"{key} must have at least {cols} columns")
            return value
        route = payload.get("route")
        trip = cls(str(payload.get("trip_id", "request")), arr("track", 2),
                   arr("nodes", 2), arr("arcs", 2),
                   None if route is None else np.asarray(route, dtype=np.int64).reshape(-1))
        trip.validate()
        return trip

    def validate(self) -> None:
        if self.track.ndim != 2 or self.track.shape[1] < 2 or not len(self.track):
            raise ValueError("track must contain at least one [longitude, latitude] point")
        if self.nodes.ndim != 2 or self.nodes.shape[1] < 2 or not len(self.nodes):
            raise ValueError("nodes must contain [longitude, latitude] rows")
        if self.arcs.ndim != 2 or self.arcs.shape[1] < 2 or not len(self.arcs):
            raise ValueError("arcs must contain [start_node_index, end_node_index] rows")
        if not np.isfinite(self.track[:, :2]).all() or not np.isfinite(self.nodes[:, :2]).all():
            raise ValueError("coordinates must be finite")
        if np.any(self.arcs[:, :2] < 0) or np.any(self.arcs[:, :2] >= len(self.nodes)):
            raise ValueError("arcs contain node indices outside the nodes array")
        if self.route is not None and (np.any(self.route < 0) or np.any(self.route >= len(self.arcs))):
            raise ValueError("route contains arc indices outside the arcs array")


class ProjectedGraph:
    """A trip-local metric projection and indexed edge geometry."""
    def __init__(self, trip: Trip):
        self.trip = trip
        self.lon0 = float(np.mean(trip.track[:, 0]))
        self.lat0 = float(np.mean(trip.track[:, 1]))
        self.cos_lat = max(0.01, math.cos(math.radians(self.lat0)))
        self.track_xy = self.xy(trip.track[:, :2])
        self.nodes_xy = self.xy(trip.nodes[:, :2])
        edge_nodes = trip.arcs[:, :2].astype(np.int64)
        self.start = self.nodes_xy[edge_nodes[:, 0]]
        self.end = self.nodes_xy[edge_nodes[:, 1]]
        self.vec = self.end - self.start
        self.length = np.linalg.norm(self.vec, axis=1).clip(0.1)
        self.heading = np.arctan2(self.vec[:, 1], self.vec[:, 0])
        self.mid = (self.start + self.end) / 2

        self.index_points = np.concatenate((self.mid, self.start, self.end), axis=0)
        self.index_edge = np.tile(np.arange(len(self.start)), 3)
        self.tree = cKDTree(self.index_points)
        self.edge_adjacency = self._make_adjacency(edge_nodes)

    def xy(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=np.float64)
        return np.column_stack(((lonlat[:, 0] - self.lon0) * (math.pi / 180) * EARTH_M * self.cos_lat,
                                (lonlat[:, 1] - self.lat0) * (math.pi / 180) * EARTH_M))

    def _make_adjacency(self, edge_nodes: np.ndarray) -> list[set[int]]:
        by_start: dict[int, list[int]] = {}
        for i, (a, _) in enumerate(edge_nodes):
            by_start.setdefault(int(a), []).append(i)
        result = []
        for _, b in edge_nodes:
            result.append(set(by_start.get(int(b), ())))
        return result

    def candidates(self, xy: np.ndarray, k: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = min(max(1, k), len(self.start))


        probe = min(max(k, 5*k), len(self.index_points))
        _, indexes = self.tree.query(xy, k=probe)
        raw = np.unique(self.index_edge[np.asarray(indexes).reshape(-1)])
        p = xy[None, :]
        v = self.vec[raw]
        t = np.sum((p - self.start[raw]) * v, axis=1) / np.sum(v * v, axis=1).clip(1e-8)
        t = np.clip(t, 0.0, 1.0)
        projected = self.start[raw] + t[:, None] * v
        dist = np.linalg.norm(projected - p, axis=1)
        order = np.argsort(dist)[:k]
        return raw[order], dist[order], t[order]

    def features(self, point_index: int, edge_ids: np.ndarray, distances: np.ndarray,
                 fractions: np.ndarray) -> np.ndarray:
        if point_index == 0:
            obs_heading = np.zeros(len(edge_ids))
            speed = np.zeros(len(edge_ids))
            movement_distance = np.zeros(len(edge_ids))
            time_gap = np.zeros(len(edge_ids))
        else:
            delta = self.track_xy[point_index] - self.track_xy[point_index - 1]
            obs_heading = np.full(len(edge_ids), math.atan2(delta[1], delta[0]))
            movement_distance = np.full(len(edge_ids), np.linalg.norm(delta))
            dt = 1.0
            if self.trip.track.shape[1] > 2:
                dt = max(0.001, float(self.trip.track[point_index, 2] - self.trip.track[point_index - 1, 2]))
            speed = np.full(len(edge_ids), np.linalg.norm(delta) / dt)
            time_gap = np.full(len(edge_ids), dt)
        angle = np.abs((self.heading[edge_ids] - obs_heading + np.pi) % (2 * np.pi) - np.pi)
        if self.trip.track.shape[1] > 2 and len(self.trip.track) > 1:
            span = max(1e-9, float(self.trip.track[-1,2]-self.trip.track[0,2]))
            progress = float((self.trip.track[point_index,2]-self.trip.track[0,2])/span)
        else:
            progress = point_index / max(1, len(self.track_xy)-1)
        return np.column_stack((distances, angle, self.length[edge_ids], fractions,
                                speed, np.full(len(edge_ids), progress), movement_distance,
                                time_gap, np.cos(angle)))

    def labels_from_route(self) -> np.ndarray:
        """Associate each GPS sample with nearby reference-route geometry.

        Indexing route segment endpoints and interior points avoids the former fixed
        +/-30-row progress window, which failed when edge lengths and trip sampling
        rates differed. A small backward-progress penalty resolves spatial ties while
        keeping the label tied to the supplied route geometry.
        """
        assert self.trip.route is not None
        route = self.trip.route
        route_start = self.start[route]
        route_vec = self.vec[route]
        route_len2 = np.sum(route_vec * route_vec, axis=1).clip(1e-8)
        fractions = np.linspace(0, 1, 5)
        route_index_points = np.concatenate([
            route_start + f * route_vec for f in fractions
        ], axis=0)
        route_index_positions = np.tile(np.arange(len(route)), len(fractions))
        route_tree = cKDTree(route_index_points)
        labels = np.full(len(self.track_xy), -1, dtype=np.int64)
        previous_position = 0
        for i, point in enumerate(self.track_xy):
            _, near_indexes = route_tree.query(point, k=min(32, len(route_index_points)))
            positions = np.unique(route_index_positions[np.asarray(near_indexes).reshape(-1)])
            v = route_vec[positions]
            t = np.clip(np.sum((point-route_start[positions])*v, axis=1)/route_len2[positions], 0, 1)
            projection = route_start[positions] + t[:, None]*v
            distances = np.linalg.norm(projection-point, axis=1)


            backward = np.maximum(0, previous_position-positions) * (25.0 / max(1, len(route)//20))
            chosen = int(np.argmin(distances + backward))
            previous_position = max(previous_position, int(positions[chosen]))
            labels[i] = int(route[positions[chosen]])
        return labels


def candidate_inputs(trip: Trip, k: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build prediction inputs from GPS and graph only; never reads reference labels."""
    graph = ProjectedGraph(trip)
    xs, ids, obs_ids = [], [], []
    for i, point in enumerate(graph.track_xy):
        edge_ids, dists, fractions = graph.candidates(point, k)
        features = graph.features(i, edge_ids, dists, fractions)
        xs.append(features)
        ids.append(edge_ids)
        obs_ids.append(np.full(len(edge_ids), i, dtype=np.int64))
    return np.vstack(xs), np.concatenate(ids), np.concatenate(obs_ids)


def candidate_examples(trip: Trip, k: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate features and separate route-derived evaluation/training labels."""
    if trip.route is None:
        raise ValueError('candidate_examples requires a reference route')
    x, ids, obs_ids = candidate_inputs(trip,k)
    labels = ProjectedGraph(trip).labels_from_route()
    y=ids==labels[obs_ids]
    return x,y,ids,obs_ids,labels


def candidate_feature_rows(trip: Trip, k: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible feature matrix, candidate labels and candidate arc IDs."""
    x, y, ids, _, _ = candidate_examples(trip, k)
    return x, y, ids


def labeled_candidate_mask(y: np.ndarray, obs_ids: np.ndarray, n_observations: int) -> np.ndarray:
    """Candidate rows from observations whose reference arc exists in the candidate set."""
    has_positive = np.bincount(obs_ids, weights=np.asarray(y, dtype=np.int8),
                               minlength=n_observations) > 0
    return has_positive[obs_ids]


def route_continuity(trip: Trip, predicted_arcs: Iterable[int]) -> float:
    """Fraction of consecutive predicted arc pairs that are connected (including repeats)."""
    graph = ProjectedGraph(trip)
    arcs = list(map(int, predicted_arcs))
    if len(arcs) < 2:
        return 1.0
    connected = sum(a == b or b in graph.edge_adjacency[a] for a, b in zip(arcs, arcs[1:]))
    return connected / (len(arcs) - 1)


def viterbi_candidate_paths(trip: Trip, arc_ids: np.ndarray, obs_ids: np.ndarray,
                           row_emissions: np.ndarray, transition_weights: Iterable[float]) -> dict[float, np.ndarray]:
    graph = ProjectedGraph(trip)
    if not len(obs_ids):
        raise ValueError('Cannot decode an empty GPS track')
    starts = np.r_[0, np.flatnonzero(obs_ids[1:] != obs_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(obs_ids)]
    candidates = [arc_ids[start:end] for start, end in zip(starts, ends)]
    emissions = [row_emissions[start:end] for start, end in zip(starts, ends)]
    bases = []
    direction_scores = []
    for t in range(1, len(candidates)):
        prev_edges, cur_edges = candidates[t-1], candidates[t]
        step = float(np.linalg.norm(graph.track_xy[t] - graph.track_xy[t-1]))
        transition = np.full((len(prev_edges), len(cur_edges)), -3.0, dtype=np.float64)
        transition.fill(-4.0)
        forward = np.zeros_like(transition, dtype=bool)
        backward = np.zeros_like(transition, dtype=bool)
        for row, edge in enumerate(prev_edges):
            forward[row] = np.isin(cur_edges, tuple(graph.edge_adjacency[int(edge)]))
        for column, edge in enumerate(cur_edges):
            backward[:, column] = np.isin(prev_edges, tuple(graph.edge_adjacency[int(edge)]))
        transition[backward] = -1.5
        transition[forward] = -0.2
        same = prev_edges[:, None] == cur_edges[None, :]
        transition[same] = -0.25
        direction = np.zeros_like(transition)
        if step > 1:
            obs_delta = graph.track_xy[t] - graph.track_xy[t-1]
            obs_norm = np.linalg.norm(obs_delta)
            mid_delta = graph.mid[cur_edges][None, :, :] - graph.mid[prev_edges][:, None, :]
            mid_norm = np.linalg.norm(mid_delta, axis=2)
            valid = mid_norm > 1
            cosine = np.einsum('ijk,k->ij', mid_delta, obs_delta) / (mid_norm * obs_norm + 1e-9)
            direction[valid] = cosine[valid]
        bases.append(transition)
        direction_scores.append(direction)

    paths = {}
    for weight in transition_weights:
        dp = emissions[0].copy()
        backpointers = []
        for t, (base, direction) in enumerate(zip(bases, direction_scores), 1):
            transition = base + float(weight) * direction
            values = dp[:, None] + transition
            bp = np.argmax(values, axis=0)
            dp = emissions[t] + values[bp, np.arange(len(candidates[t]))]
            backpointers.append(bp)
        state = int(np.argmax(dp))
        states = [state]
        for bp in reversed(backpointers):
            state = int(bp[state])
            states.append(state)
        states.reverse()
        paths[float(weight)] = np.array([candidates[t][s] for t, s in enumerate(states)], dtype=np.int64)
    return paths


def network_viterbi_paths(trip: Trip, arc_ids: np.ndarray, obs_ids: np.ndarray,
                          fractions: np.ndarray, row_emissions: np.ndarray,
                          settings: Iterable[tuple[float, float]]) -> dict[tuple[float, float], np.ndarray]:
    graph = ProjectedGraph(trip)
    settings = list(settings)
    edge_nodes = trip.arcs[:, :2].astype(np.int64)
    edge_lengths = graph.length
    edge_weights = {}
    for edge, (start, end) in enumerate(edge_nodes):
        key = (int(start), int(end))
        edge_weights[key] = min(edge_weights.get(key, float('inf')), float(edge_lengths[edge]))
    keys = list(edge_weights)
    network = csr_matrix((np.fromiter((edge_weights[key] for key in keys), dtype=np.float64),
                          (np.fromiter((key[0] for key in keys), dtype=np.int64),
                           np.fromiter((key[1] for key in keys), dtype=np.int64))),
                         shape=(len(trip.nodes), len(trip.nodes)))
    starts = np.r_[0, np.flatnonzero(obs_ids[1:] != obs_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(obs_ids)]
    candidates = [arc_ids[start:end] for start, end in zip(starts, ends)]
    fractions_by_observation = [fractions[start:end] for start, end in zip(starts, ends)]
    emissions = [row_emissions[start:end] for start, end in zip(starts, ends)]
    unique_sources = np.unique(edge_nodes[arc_ids, 1])
    source_rows = dijkstra(network, directed=True, indices=unique_sources)
    if source_rows.ndim == 1:
        source_rows = source_rows[None, :]
    source_lookup = {int(node): row for row, node in enumerate(unique_sources)}
    transition_errors = []
    for t in range(1, len(candidates)):
        prev_edges = candidates[t - 1]
        cur_edges = candidates[t]
        prev_fraction = fractions_by_observation[t - 1]
        cur_fraction = fractions_by_observation[t]
        source_ids = edge_nodes[prev_edges, 1]
        target_ids = edge_nodes[cur_edges, 0]
        distances = source_rows[np.asarray([source_lookup[int(node)] for node in source_ids])[:, None],
                                target_ids[None, :]]
        network_distance = ((1.0 - prev_fraction[:, None]) * edge_lengths[prev_edges, None]
                            + distances
                            + cur_fraction[None, :] * edge_lengths[None, cur_edges])
        same_forward = ((prev_edges[:, None] == cur_edges[None, :])
                        & (cur_fraction[None, :] >= prev_fraction[:, None]))
        same_distance = ((cur_fraction[None, :] - prev_fraction[:, None])
                         * edge_lengths[prev_edges, None])
        network_distance[same_forward] = same_distance[same_forward]
        observed_distance = float(np.linalg.norm(graph.track_xy[t] - graph.track_xy[t - 1]))
        error = np.abs(network_distance - observed_distance)
        transition_errors.append(error)
    paths = {}
    for distance_scale, transition_weight in settings:
        dp = emissions[0].copy()
        backpointers = []
        for t, error in enumerate(transition_errors, 1):
            transition = -error / max(1.0, distance_scale)
            transition[~np.isfinite(transition)] = -20.0
            transition = np.maximum(transition, -20.0) * transition_weight
            values = dp[:, None] + transition
            best_previous = np.argmax(values, axis=0)
            dp = emissions[t] + values[best_previous, np.arange(len(candidates[t]))]
            backpointers.append(best_previous)
        state = int(np.argmax(dp))
        states = [state]
        for backpointer in reversed(backpointers):
            state = int(backpointer[state])
            states.append(state)
        states.reverse()
        paths[(float(distance_scale), float(transition_weight))] = np.asarray(
            [candidates[t][state] for t, state in enumerate(states)], dtype=np.int64)
    return paths


def network_viterbi_path(trip: Trip, arc_ids: np.ndarray, obs_ids: np.ndarray,
                         fractions: np.ndarray, row_emissions: np.ndarray,
                         distance_scale: float, transition_weight: float) -> np.ndarray:
    return network_viterbi_paths(trip, arc_ids, obs_ids, fractions, row_emissions,
                                 [(distance_scale, transition_weight)])[(float(distance_scale),
                                                                         float(transition_weight))]


def decode(trip: Trip, model: Any, candidate_count: int = 12,
           transition_weight: float = 1.5) -> dict[str, Any]:
    graph = ProjectedGraph(trip)
    features, arc_ids, obs_ids = candidate_inputs(trip, candidate_count)
    probabilities = model.predict_proba(features)
    classes = list(model.classes_) if hasattr(model, 'classes_') else list(model[-1].classes_)
    if 1 in classes:
        score = probabilities[:, classes.index(1)]
    else:
        score = np.exp(-features[:, 0] / 20.0)
    row_emissions = np.log(np.maximum(score, 1e-9)) - features[:, 0] / 60.0
    matched = viterbi_candidate_paths(trip, arc_ids, obs_ids, row_emissions,
                                     [transition_weight])[float(transition_weight)]

    route = [int(matched[0])]
    for e in matched[1:]:
        if int(e) != route[-1]:
            route.append(int(e))
    return {
        "trip_id": trip.trip_id,
        "route": route,
        "matched_arc_indices": matched.tolist(),
        "matched_coordinates": graph.trip.nodes[graph.trip.arcs[matched, :2].astype(int)].mean(axis=1).tolist(),
        "candidate_count": candidate_count,
    }
