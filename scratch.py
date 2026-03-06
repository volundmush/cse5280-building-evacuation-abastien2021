"""Gradient-descent evacuation simulator for small multi-floor buildings.

The simulator builds a scalar potential field over each walkable floor surface and
connects floors through ramps. Agents descend that field with additional
repulsion from fixtures and other agents.

Supported floorplan schema (new style)
--------------------------------------
{
  "name": "Three story demo",
  "grid": {"cell_size": 0.35, "padding": 1.0},
  "floors": [
    {
      "name": "Ground",
      "z": 0.0,
      "thickness": 0.18,
      "polygon": [[0,0], [18,0], [18,12], [0,12]],
      "obstacles": [
        {"type": "box", "polygon": [[6,4], [7.4,4], [7.4,5.8], [6,5.8]], "height": 1.2},
        {"type": "column", "center": [12,8], "radius": 0.45, "height": 3.0}
      ]
    }
  ],
  "fixtures": [
    {"type": "wall", "polygon": [[0,0], [18,0], [18,0.25], [0,0.25]], "z": [0, 9]}
  ],
  "ramps": [
    {
      "name": "upper_to_mid",
      "polygon": [[14,7], [18,7], [18,9], [14,9]],
      "start": [18,8,6.09],
      "end": [14,8,3.09],
      "width": 2.0,
      "thickness": 0.16,
      "from_floor": 2,
      "to_floor": 1
    }
  ],
  "openings": [{"min": [13.8, 6.9, 5.8], "max": [18.2, 9.1, 6.5]}],
  "goal": {"min": [-0.5, 5.2, -0.1], "max": [0.8, 6.8, 1.0]}
}

Scenario schema (new style)
---------------------------
{
  "name": "Mixed agents",
  "simulation": {"dt": 0.05, "steps": 700, "seed": 7},
  "agent_defaults": {
    "radius": 0.22,
    "max_speed": 1.35,
    "nav_gain": 3.0,
    "wall_strength": 5.5,
    "wall_range": 1.0,
    "social_strength": 1.8,
    "social_range": 1.1,
    "ramp_gain": 3.6,
    "goal_gain": 2.2,
    "damping": 0.72
  },
  "agents": [
    {
      "count": 20,
      "floor": 2,
      "offset_range": {"x": [-0.75, 0.75], "y": [-0.65, 0.65]},
      "color": "royalblue",
      "repulsion": {
        "strength": 1.8,
        "range": 1.0,
        "anisotropy": {
          "enabled": true,
          "forward_strength": 1.7,
          "backward_strength": 0.55,
          "forward_angle_deg": 80
        }
      }
    }
  ],
  "obstacles": [
    {"type": "box", "floor": 1, "offset": [0.15, -0.2], "size": [1.1, 1.1], "height": 1.2}
  ]
}

Legacy fields from old_scratch.py are also accepted where practical.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Vector = np.ndarray
Polygon2D = np.ndarray

DATA_FOLDER = pathlib.Path(__file__).resolve().parent / "data"
EPS = 1e-9
INF = 1e18


@dataclass
class Opening:
    min_corner: Vector
    max_corner: Vector


@dataclass
class Goal:
    min_corner: Vector
    max_corner: Vector
    color: str = "green4"

    @property
    def center(self) -> Vector:
        return 0.5 * (self.min_corner + self.max_corner)


@dataclass
class Floor:
    index: int
    name: str
    z: float
    thickness: float
    polygon: Polygon2D
    color: str = "lightgray"
    obstacles: List["Fixture"] = field(default_factory=list)

    @property
    def surface_z(self) -> float:
        return self.z + 0.5 * self.thickness

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return polygon_bbox(self.polygon)


@dataclass
class Fixture:
    kind: str
    polygon: Optional[Polygon2D] = None
    center: Optional[Vector] = None
    radius: Optional[float] = None
    base_z: float = 0.0
    top_z: float = 0.0
    color: str = "slategray"
    alpha: float = 0.25

    def active_at_z(self, z: float) -> bool:
        return self.base_z - EPS <= z <= self.top_z + EPS


@dataclass
class Ramp:
    name: str
    polygon: Polygon2D
    start: Vector
    end: Vector
    width: float
    from_floor: int
    to_floor: int
    thickness: float = 0.16
    color: str = "orange5"

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def centerline_xy(self) -> Tuple[Vector, Vector]:
        return self.start[:2].copy(), self.end[:2].copy()

    @property
    def lower_floor(self) -> int:
        return self.from_floor if self.start[2] < self.end[2] else self.to_floor

    @property
    def upper_floor(self) -> int:
        return self.from_floor if self.start[2] > self.end[2] else self.to_floor


@dataclass
class RepulsionConfig:
    strength: float = 1.6
    range: float = 1.0
    anisotropy_enabled: bool = False
    forward_strength: float = 1.6
    backward_strength: float = 0.6
    forward_angle_deg: float = 80.0


@dataclass
class AgentConfig:
    radius: float = 0.22
    max_speed: float = 1.35
    nav_gain: float = 3.0
    wall_strength: float = 5.5
    wall_range: float = 1.0
    social_strength: float = 1.8
    social_range: float = 1.1
    ramp_gain: float = 3.6
    goal_gain: float = 2.2
    damping: float = 0.72


@dataclass
class SimulationConfig:
    dt: float = 0.05
    steps: int = 700
    seed: int = 7
    cell_size: float = 0.35
    offscreen: bool = False


@dataclass
class ScenarioObstacleSpec:
    raw: Dict


@dataclass
class AgentGroup:
    count: int
    floor: int
    color: str
    repulsion: RepulsionConfig
    placement: Dict


@dataclass
class Scenario:
    name: str
    simulation: SimulationConfig
    agent_defaults: AgentConfig
    groups: List[AgentGroup]
    obstacles: List[ScenarioObstacleSpec]


@dataclass
class Floorplan:
    name: str
    floors: List[Floor]
    fixtures: List[Fixture]
    ramps: List[Ramp]
    openings: List[Opening]
    goal: Goal
    bounds_min: Vector
    bounds_max: Vector


@dataclass
class Agent:
    pos: Vector
    vel: Vector
    floor_index: int
    radius: float
    color: str
    repulsion: RepulsionConfig
    active: bool = True
    reached_goal: bool = False
    ramp_name: Optional[str] = None


@dataclass
class PortalEdge:
    target_floor: int
    target_y: int
    target_x: int
    cost: float


@dataclass
class NavigationField:
    cell_size: float
    x_coords: np.ndarray
    y_coords: np.ndarray
    walkable: np.ndarray
    distance_to_block: np.ndarray
    navigation: np.ndarray
    portal_edges: Dict[Tuple[int, int, int], List[PortalEdge]]
    goal_floor: int
    floor_lookup: Dict[int, Floor]


def vec(values: Sequence[float]) -> Vector:
    return np.array(values, dtype=float)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def norm(v: Vector) -> float:
    return float(np.linalg.norm(v))


def unit(v: Vector) -> Vector:
    n = norm(v)
    if n < EPS:
        return np.zeros_like(v, dtype=float)
    return v / n


def normalize_percent(value: float) -> float:
    if abs(value) > 1.0 and abs(value) <= 100.0:
        return value / 100.0
    return value


def polygon_bbox(polygon: Polygon2D) -> Tuple[float, float, float, float]:
    return (
        float(np.min(polygon[:, 0])),
        float(np.max(polygon[:, 0])),
        float(np.min(polygon[:, 1])),
        float(np.max(polygon[:, 1])),
    )


def polygon_area(polygon: Polygon2D) -> float:
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_polygon(raw: Sequence[Sequence[float]]) -> Polygon2D:
    polygon = np.array(raw, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("Polygons must have shape (n, 2) with n >= 3.")
    if polygon_area(polygon) < 0:
        polygon = polygon[::-1].copy()
    return polygon


def point_in_polygon(point_xy: Sequence[float], polygon: Polygon2D) -> bool:
    x, y = float(point_xy[0]), float(point_xy[1])
    inside = False
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        if ((y1 > y) != (y0 > y)) and (
            x < (x0 - x1) * (y - y1) / ((y0 - y1) + EPS) + x1
        ):
            inside = not inside
        x0, y0 = x1, y1
    return inside


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, polygon: Polygon2D) -> np.ndarray:
    inside = np.zeros(xs.shape, dtype=bool)
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        intersects = ((y1 > ys) != (y0 > ys)) & (
            xs < (x0 - x1) * (ys - y1) / ((y0 - y1) + EPS) + x1
        )
        inside ^= intersects
        x0, y0 = x1, y1
    return inside


def closest_point_on_segment_2d(point_xy: Vector, a_xy: Vector, b_xy: Vector) -> Tuple[Vector, float]:
    ab = b_xy - a_xy
    denom = float(np.dot(ab, ab))
    if denom < EPS:
        return a_xy.copy(), 0.0
    t = clamp(float(np.dot(point_xy - a_xy, ab)) / denom, 0.0, 1.0)
    return a_xy + t * ab, t


def distance_point_to_aabb(point: Vector, min_corner: Vector, max_corner: Vector) -> float:
    clamped = np.array(
        [
            clamp(point[0], min_corner[0], max_corner[0]),
            clamp(point[1], min_corner[1], max_corner[1]),
            clamp(point[2], min_corner[2], max_corner[2]),
        ],
        dtype=float,
    )
    return norm(point - clamped)


def aabb_contains(point: Vector, min_corner: Vector, max_corner: Vector) -> bool:
    return bool(np.all(point >= min_corner - EPS) and np.all(point <= max_corner + EPS))


def triangulate_polygon(polygon: Polygon2D) -> List[Tuple[int, int, int]]:
    polygon = ensure_polygon(polygon)
    remaining = list(range(len(polygon)))
    triangles: List[Tuple[int, int, int]] = []

    def is_convex(i0: int, i1: int, i2: int) -> bool:
        a = polygon[i1] - polygon[i0]
        b = polygon[i2] - polygon[i1]
        return float(a[0] * b[1] - a[1] * b[0]) > EPS

    def point_in_triangle(p: Vector, a: Vector, b: Vector, c: Vector) -> bool:
        v0 = c - a
        v1 = b - a
        v2 = p - a
        dot00 = float(np.dot(v0, v0))
        dot01 = float(np.dot(v0, v1))
        dot02 = float(np.dot(v0, v2))
        dot11 = float(np.dot(v1, v1))
        dot12 = float(np.dot(v1, v2))
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < EPS:
            return False
        inv = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        return u >= -EPS and v >= -EPS and (u + v) <= 1.0 + EPS

    guard = 0
    while len(remaining) > 3 and guard < 10000:
        guard += 1
        ear_found = False
        for idx in range(len(remaining)):
            i0 = remaining[(idx - 1) % len(remaining)]
            i1 = remaining[idx]
            i2 = remaining[(idx + 1) % len(remaining)]
            if not is_convex(i0, i1, i2):
                continue
            a, b, c = polygon[i0], polygon[i1], polygon[i2]
            if any(
                point_in_triangle(polygon[j], a, b, c)
                for j in remaining
                if j not in (i0, i1, i2)
            ):
                continue
            triangles.append((i0, i1, i2))
            del remaining[idx]
            ear_found = True
            break
        if not ear_found:
            break
    if len(remaining) == 3:
        triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def build_prism_mesh(polygon: Polygon2D, z0: float, z1: float) -> Tuple[List[List[float]], List[List[int]]]:
    polygon = ensure_polygon(polygon)
    triangles = triangulate_polygon(polygon)
    n = len(polygon)
    vertices: List[List[float]] = []
    for x, y in polygon:
        vertices.append([float(x), float(y), float(z0)])
    for x, y in polygon:
        vertices.append([float(x), float(y), float(z1)])

    faces: List[List[int]] = []
    for a, b, c in triangles:
        faces.append([a, c, b])
        faces.append([a + n, b + n, c + n])
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    return vertices, faces


def build_surface_mesh(polygon: Polygon2D, z: float) -> Tuple[List[List[float]], List[List[int]]]:
    polygon = ensure_polygon(polygon)
    triangles = triangulate_polygon(polygon)
    vertices = [[float(x), float(y), float(z)] for x, y in polygon]
    faces = [[a, b, c] for a, b, c in triangles]
    return vertices, faces


def build_ramp_mesh(ramp: Ramp) -> Tuple[List[List[float]], List[List[int]]]:
    polygon = ensure_polygon(ramp.polygon)
    triangles = triangulate_polygon(polygon)
    top_vertices: List[List[float]] = []
    bottom_vertices: List[List[float]] = []
    for x, y in polygon:
        z = ramp_height_at_xy(ramp, np.array([x, y], dtype=float))
        top_vertices.append([float(x), float(y), float(z + 0.5 * ramp.thickness)])
        bottom_vertices.append([float(x), float(y), float(z - 0.5 * ramp.thickness)])
    vertices = bottom_vertices + top_vertices
    n = len(polygon)
    faces: List[List[int]] = []
    for a, b, c in triangles:
        faces.append([a, c, b])
        faces.append([a + n, b + n, c + n])
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    return vertices, faces


def make_rect_polygon(min_corner: Sequence[float], max_corner: Sequence[float]) -> Polygon2D:
    x0, y0 = float(min_corner[0]), float(min_corner[1])
    x1, y1 = float(max_corner[0]), float(max_corner[1])
    return ensure_polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def rectangle_around_segment(start_xy: Vector, end_xy: Vector, width: float) -> Polygon2D:
    d = end_xy - start_xy
    d_unit = unit(d)
    if norm(d_unit) < EPS:
        d_unit = np.array([1.0, 0.0], dtype=float)
    normal = np.array([-d_unit[1], d_unit[0]], dtype=float)
    half_w = 0.5 * width
    return ensure_polygon(
        [
            start_xy - half_w * normal,
            end_xy - half_w * normal,
            end_xy + half_w * normal,
            start_xy + half_w * normal,
        ]
    )


def load_json_from_folder(folder: pathlib.Path, value: str) -> Dict:
    candidate = pathlib.Path(value)
    if candidate.suffix == ".json" and candidate.exists():
        path = candidate
    elif candidate.exists():
        path = candidate
    else:
        name = value[:-5] if value.endswith(".json") else value
        path = folder / f"{name}.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def nearest_floor_index(z_value: float, floors: Sequence[Floor]) -> int:
    return min(range(len(floors)), key=lambda idx: abs(floors[idx].surface_z - z_value))


def parse_goal(data: Dict) -> Goal:
    if "min" in data and "max" in data:
        return Goal(vec(data["min"]), vec(data["max"]), color=data.get("color", "green4"))
    center = vec(data.get("center", [0.0, 0.0, 0.0]))
    radius = float(data.get("radius", 0.6))
    return Goal(center - radius, center + radius, color=data.get("color", "green4"))


def parse_opening(data: Dict) -> Opening:
    return Opening(vec(data["min"]), vec(data["max"]))


def parse_fixture(data: Dict, default_base: float, default_top: float) -> Fixture:
    kind = data.get("type", data.get("kind", "box")).lower()
    color = data.get("color", "slategray")
    alpha = float(data.get("alpha", 0.28))
    if "min" in data and "max" in data:
        min_corner = vec(data["min"])
        max_corner = vec(data["max"])
        return Fixture(
            kind=kind,
            polygon=make_rect_polygon(min_corner[:2], max_corner[:2]),
            base_z=float(min_corner[2]),
            top_z=float(max_corner[2]),
            color=color,
            alpha=alpha,
        )

    if kind in {"column", "pillar"}:
        center_xy = np.array(data.get("center", [0.0, 0.0])[:2], dtype=float)
        base_z = float(data.get("base_z", data.get("z", default_base)))
        top_z = float(data.get("top_z", base_z + data.get("height", default_top - default_base)))
        return Fixture(
            kind="column",
            center=center_xy,
            radius=float(data.get("radius", 0.5)),
            base_z=base_z,
            top_z=top_z,
            color=color,
            alpha=alpha,
        )

    polygon = ensure_polygon(data["polygon"])
    z_range = data.get("z")
    if z_range is not None:
        base_z = float(z_range[0])
        top_z = float(z_range[1])
    else:
        base_z = float(data.get("base_z", data.get("z0", default_base)))
        top_z = float(data.get("top_z", data.get("z1", base_z + data.get("height", default_top - default_base))))
    return Fixture(
        kind=kind,
        polygon=polygon,
        base_z=base_z,
        top_z=top_z,
        color=color,
        alpha=alpha,
    )


def parse_ramp(data: Dict, floors: Sequence[Floor]) -> Ramp:
    if "start" in data and "end" in data:
        start = vec(data["start"])
        end = vec(data["end"])
    else:
        start_xy = vec(data["from_xy"])
        end_xy = vec(data["to_xy"])
        from_floor = int(data["from_floor"])
        to_floor = int(data["to_floor"])
        start = np.array([start_xy[0], start_xy[1], floors[from_floor].surface_z], dtype=float)
        end = np.array([end_xy[0], end_xy[1], floors[to_floor].surface_z], dtype=float)

    width = float(data.get("width", 2.0))
    polygon = ensure_polygon(data["polygon"]) if "polygon" in data else rectangle_around_segment(start[:2], end[:2], width)
    from_floor = int(data.get("from_floor", nearest_floor_index(start[2], floors)))
    to_floor = int(data.get("to_floor", nearest_floor_index(end[2], floors)))
    return Ramp(
        name=data.get("name", f"ramp_{from_floor}_{to_floor}"),
        polygon=polygon,
        start=start,
        end=end,
        width=width,
        from_floor=from_floor,
        to_floor=to_floor,
        thickness=float(data.get("thickness", 0.16)),
        color=data.get("color", "orange5"),
    )


def load_floorplan(spec: str) -> Floorplan:
    data = load_json_from_folder(DATA_FOLDER / "floorplans", spec)
    name = data.get("name", pathlib.Path(spec).stem)

    floors: List[Floor] = []
    raw_floors = data.get("floors", [])
    if not raw_floors:
        raise ValueError("Floorplan requires a non-empty 'floors' array.")
    for idx, raw_floor in enumerate(raw_floors):
        polygon = raw_floor.get("polygon")
        if polygon is None:
            bounds = data.get("bounds")
            if bounds is None:
                raise ValueError("Each floor needs a polygon unless global bounds are supplied.")
            polygon = [
                [bounds["x"][0], bounds["y"][0]],
                [bounds["x"][1], bounds["y"][0]],
                [bounds["x"][1], bounds["y"][1]],
                [bounds["x"][0], bounds["y"][1]],
            ]
        floor = Floor(
            index=idx,
            name=raw_floor.get("name", f"floor_{idx}"),
            z=float(raw_floor.get("z", idx * 3.0)),
            thickness=float(raw_floor.get("thickness", 0.18)),
            polygon=ensure_polygon(polygon),
            color=raw_floor.get("color", "lightgray"),
        )
        floors.append(floor)

    z_min = min(f.z for f in floors)
    z_max = max(f.z + f.thickness for f in floors)
    fixtures: List[Fixture] = []
    for raw in data.get("fixtures", []):
        fixtures.append(parse_fixture(raw, z_min, z_max + 3.0))
    for raw in data.get("walls", []):
        fixtures.append(parse_fixture({**raw, "type": raw.get("type", "wall")}, z_min, z_max + 3.0))
    for raw in data.get("obstacles", []):
        fixtures.append(parse_fixture(raw, z_min, z_max + 3.0))
    for floor, raw_floor in zip(floors, raw_floors):
        for raw in raw_floor.get("obstacles", []):
            floor.obstacles.append(
                parse_fixture(
                    raw,
                    floor.z,
                    floor.z + raw.get("height", floor.thickness + 1.5),
                )
            )

    openings = [parse_opening(raw) for raw in data.get("openings", data.get("repulsion_masks", []))]
    ramps = [parse_ramp(raw, floors) for raw in data.get("ramps", data.get("stairs", []))]
    goal = parse_goal(data.get("goal", {"center": [0.0, 0.0, floors[0].surface_z], "radius": 0.75}))

    all_polygons = [floor.polygon for floor in floors]
    all_polygons.extend(fixture.polygon for fixture in fixtures if fixture.polygon is not None)
    all_polygons.extend(ramp.polygon for ramp in ramps)
    xmins, xmaxs, ymins, ymaxs = zip(*(polygon_bbox(poly) for poly in all_polygons))
    bounds_min = np.array([
        min(xmins) - 1.0,
        min(ymins) - 1.0,
        min([goal.min_corner[2], z_min]) - 0.5,
    ], dtype=float)
    bounds_max = np.array([
        max(xmaxs) + 1.0,
        max(ymaxs) + 1.0,
        max([goal.max_corner[2], z_max]) + 1.0,
    ], dtype=float)

    return Floorplan(
        name=name,
        floors=floors,
        fixtures=fixtures,
        ramps=ramps,
        openings=openings,
        goal=goal,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def parse_repulsion(data: Dict) -> RepulsionConfig:
    anis = data.get("anisotropy", {})
    mode = str(data.get("mode", "")).lower()
    return RepulsionConfig(
        strength=float(data.get("strength", 1.6)),
        range=float(data.get("range", 1.0)),
        anisotropy_enabled=bool(anis.get("enabled", mode == "anisotropic")),
        forward_strength=float(anis.get("forward_strength", 1.6)),
        backward_strength=float(anis.get("backward_strength", 0.6)),
        forward_angle_deg=float(anis.get("forward_angle_deg", 80.0)),
    )


def parse_agent_defaults(data: Dict) -> AgentConfig:
    if not data:
        return AgentConfig()
    legacy = dict(data)
    return AgentConfig(
        radius=float(legacy.get("radius", 0.22)),
        max_speed=float(legacy.get("max_speed", 1.35)),
        nav_gain=float(legacy.get("nav_gain", legacy.get("goal_attraction", 3.0))),
        wall_strength=float(legacy.get("wall_strength", legacy.get("wall_repulsion", 5.5))),
        wall_range=float(legacy.get("wall_range", 1.0)),
        social_strength=float(legacy.get("social_strength", 1.8)),
        social_range=float(legacy.get("social_range", 1.1)),
        ramp_gain=float(legacy.get("ramp_gain", legacy.get("ramp_attraction", 3.6))),
        goal_gain=float(legacy.get("goal_gain", 2.2)),
        damping=float(legacy.get("damping", 0.72)),
    )


def load_scenario(spec: str, floorplan: Floorplan) -> Scenario:
    data = load_json_from_folder(DATA_FOLDER / "scenarios", spec)
    simulation_raw = data.get("simulation", {})
    grid_raw = data.get("grid", {})
    simulation = SimulationConfig(
        dt=float(simulation_raw.get("dt", 0.05)),
        steps=int(simulation_raw.get("steps", 700)),
        seed=int(simulation_raw.get("seed", 7)),
        cell_size=float(grid_raw.get("cell_size", simulation_raw.get("cell_size", 0.35))),
        offscreen=bool(simulation_raw.get("offscreen", False)),
    )
    defaults = parse_agent_defaults(data.get("agent_defaults", data.get("agent", {})))

    raw_groups = data.get("agents", data.get("evacuees", []))
    groups: List[AgentGroup] = []
    for raw in raw_groups:
        if "spawn" in raw and "box" in raw.get("spawn", {}):
            spawn_box = raw["spawn"]["box"]
            center = 0.5 * (vec(spawn_box["min"]) + vec(spawn_box["max"]))
            floor_idx = nearest_floor_index(center[2], floorplan.floors)
            floor = floorplan.floors[floor_idx]
            min_box = vec(spawn_box["min"])
            max_box = vec(spawn_box["max"])
            fx0, fx1, fy0, fy1 = floor.bbox
            placement = {
                "floor": floor_idx,
                "absolute_box": {"min": min_box.tolist(), "max": max_box.tolist()},
                "offset_range": {
                    "x": [2.0 * (min_box[0] - 0.5 * (fx0 + fx1)) / max(fx1 - fx0, EPS), 2.0 * (max_box[0] - 0.5 * (fx0 + fx1)) / max(fx1 - fx0, EPS)],
                    "y": [2.0 * (min_box[1] - 0.5 * (fy0 + fy1)) / max(fy1 - fy0, EPS), 2.0 * (max_box[1] - 0.5 * (fy0 + fy1)) / max(fy1 - fy0, EPS)],
                },
            }
        else:
            floor_idx = int(raw.get("floor", 0))
            placement = dict(raw)
            placement["floor"] = floor_idx
        groups.append(
            AgentGroup(
                count=int(raw.get("count", 1)),
                floor=floor_idx,
                color=raw.get("color", random.choice(["royalblue", "tomato", "gold", "orchid", "cyan4"])),
                repulsion=parse_repulsion(raw.get("repulsion", {})),
                placement=placement,
            )
        )

    obstacles = [ScenarioObstacleSpec(raw) for raw in data.get("obstacles", [])]
    return Scenario(
        name=data.get("name", pathlib.Path(spec).stem),
        simulation=simulation,
        agent_defaults=defaults,
        groups=groups,
        obstacles=obstacles,
    )


def scenario_obstacle_to_fixture(spec: ScenarioObstacleSpec, floorplan: Floorplan) -> Fixture:
    raw = spec.raw
    floor_index = int(raw.get("floor", 0))
    floor = floorplan.floors[floor_index]
    floor_center = np.array([
        0.5 * (floor.bbox[0] + floor.bbox[1]),
        0.5 * (floor.bbox[2] + floor.bbox[3]),
    ])
    extents = np.array([floor.bbox[1] - floor.bbox[0], floor.bbox[3] - floor.bbox[2]], dtype=float)

    if raw.get("type", "box").lower() in {"column", "pillar"}:
        offset = np.array([normalize_percent(v) for v in raw.get("offset", [0.0, 0.0])], dtype=float)
        center_xy = floor_center + 0.5 * extents * offset
        base_z = floor.z
        top_z = floor.z + float(raw.get("height", 2.5))
        return Fixture(
            kind="column",
            center=center_xy,
            radius=float(raw.get("radius", 0.45)),
            base_z=base_z,
            top_z=top_z,
            color=raw.get("color", "tan"),
            alpha=float(raw.get("alpha", 0.35)),
        )

    size = np.array(raw.get("size", [1.0, 1.0]), dtype=float)
    offset = np.array([normalize_percent(v) for v in raw.get("offset", [0.0, 0.0])], dtype=float)
    center_xy = floor_center + 0.5 * extents * offset
    min_xy = center_xy - 0.5 * size
    max_xy = center_xy + 0.5 * size
    base_z = floor.z
    top_z = floor.z + float(raw.get("height", 1.2))
    return Fixture(
        kind="box",
        polygon=make_rect_polygon(min_xy, max_xy),
        base_z=base_z,
        top_z=top_z,
        color=raw.get("color", "brown"),
        alpha=float(raw.get("alpha", 0.35)),
    )


def apply_scenario_obstacles(floorplan: Floorplan, scenario: Scenario) -> None:
    for spec in scenario.obstacles:
        floorplan.fixtures.append(scenario_obstacle_to_fixture(spec, floorplan))


def opening_contains_xy(opening: Opening, point_xy: Vector, z_value: float) -> bool:
    return (
        opening.min_corner[0] - EPS <= point_xy[0] <= opening.max_corner[0] + EPS
        and opening.min_corner[1] - EPS <= point_xy[1] <= opening.max_corner[1] + EPS
        and opening.min_corner[2] - EPS <= z_value <= opening.max_corner[2] + EPS
    )


def point_blocked_by_fixture(point_xy: Vector, z_value: float, fixture: Fixture) -> bool:
    if not fixture.active_at_z(z_value):
        return False
    if fixture.kind == "column":
        assert fixture.center is not None and fixture.radius is not None
        return norm(point_xy - fixture.center) <= fixture.radius + EPS
    if fixture.polygon is None:
        return False
    return point_in_polygon(point_xy, fixture.polygon)


def is_xy_walkable_on_floor(floorplan: Floorplan, floor_index: int, point_xy: Vector) -> bool:
    floor = floorplan.floors[floor_index]
    if not point_in_polygon(point_xy, floor.polygon):
        return False

    sample_z = floor.surface_z
    fixtures = list(floorplan.fixtures) + list(floor.obstacles)
    for fixture in fixtures:
        if not point_blocked_by_fixture(point_xy, sample_z, fixture):
            continue
        if any(opening_contains_xy(opening, point_xy, sample_z) for opening in floorplan.openings):
            continue
        return False
    return True


def floor_walkable_mask(
    floor: Floor,
    floorplan: Floorplan,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    xx, yy = np.meshgrid(x_coords, y_coords)
    walkable = points_in_polygon(xx, yy, floor.polygon)
    sample_z = floor.surface_z

    for fixture in list(floorplan.fixtures) + list(floor.obstacles):
        if fixture.kind == "column":
            if not fixture.active_at_z(sample_z):
                continue
            cx, cy = float(fixture.center[0]), float(fixture.center[1])
            rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            blocked = rr <= float(fixture.radius) + EPS
        else:
            if fixture.polygon is None or not fixture.active_at_z(sample_z):
                continue
            blocked = points_in_polygon(xx, yy, fixture.polygon)

        if floorplan.openings:
            opening_mask = np.zeros_like(blocked)
            for opening in floorplan.openings:
                opening_mask |= (
                    (xx >= opening.min_corner[0] - EPS)
                    & (xx <= opening.max_corner[0] + EPS)
                    & (yy >= opening.min_corner[1] - EPS)
                    & (yy <= opening.max_corner[1] + EPS)
                    & (sample_z >= opening.min_corner[2] - EPS)
                    & (sample_z <= opening.max_corner[2] + EPS)
                )
            blocked &= ~opening_mask
        walkable &= ~blocked
    return walkable


def nearest_walkable_index(mask: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, xy: Vector) -> Optional[Tuple[int, int]]:
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return None
    coords = np.column_stack((x_coords[indices[:, 1]], y_coords[indices[:, 0]]))
    distances = np.sum((coords - xy[None, :]) ** 2, axis=1)
    best = int(np.argmin(distances))
    return int(indices[best, 0]), int(indices[best, 1])


def nearest_walkable_xy(field: NavigationField, floor_index: int, xy: Vector) -> Vector:
    ij = nearest_walkable_index(field.walkable[floor_index], field.x_coords, field.y_coords, xy)
    if ij is None:
        return xy.copy()
    iy, ix = ij
    return np.array([field.x_coords[ix], field.y_coords[iy]], dtype=float)


def sample_scalar(grid: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, xy: Vector) -> float:
    x = clamp(float(xy[0]), float(x_coords[0]), float(x_coords[-1]))
    y = clamp(float(xy[1]), float(y_coords[0]), float(y_coords[-1]))
    ix_hi = int(np.searchsorted(x_coords, x))
    iy_hi = int(np.searchsorted(y_coords, y))
    ix1 = min(max(ix_hi, 1), len(x_coords) - 1)
    iy1 = min(max(iy_hi, 1), len(y_coords) - 1)
    ix0 = ix1 - 1
    iy0 = iy1 - 1
    x0, x1 = x_coords[ix0], x_coords[ix1]
    y0, y1 = y_coords[iy0], y_coords[iy1]
    tx = 0.0 if abs(x1 - x0) < EPS else (x - x0) / (x1 - x0)
    ty = 0.0 if abs(y1 - y0) < EPS else (y - y0) / (y1 - y0)
    v00 = float(grid[iy0, ix0])
    v10 = float(grid[iy0, ix1])
    v01 = float(grid[iy1, ix0])
    v11 = float(grid[iy1, ix1])
    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11


def sample_gradient(grid: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, xy: Vector, step: float) -> Vector:
    dx = np.array([step, 0.0], dtype=float)
    dy = np.array([0.0, step], dtype=float)
    gx = (sample_scalar(grid, x_coords, y_coords, xy + dx) - sample_scalar(grid, x_coords, y_coords, xy - dx)) / (2.0 * step)
    gy = (sample_scalar(grid, x_coords, y_coords, xy + dy) - sample_scalar(grid, x_coords, y_coords, xy - dy)) / (2.0 * step)
    return np.array([gx, gy], dtype=float)


def ramp_height_at_xy(ramp: Ramp, point_xy: Vector) -> float:
    _, t = closest_point_on_segment_2d(point_xy, ramp.start[:2], ramp.end[:2])
    return float(ramp.start[2] + t * (ramp.end[2] - ramp.start[2]))


def ramp_tangent_xy(ramp: Ramp, toward_lower: bool = True) -> Vector:
    high = ramp.start if ramp.start[2] >= ramp.end[2] else ramp.end
    low = ramp.end if ramp.start[2] >= ramp.end[2] else ramp.start
    tangent = low[:2] - high[:2] if toward_lower else high[:2] - low[:2]
    return unit(tangent)


def get_ramp_entry_exit(ramp: Ramp) -> Tuple[Tuple[int, Vector], Tuple[int, Vector]]:
    if ramp.start[2] >= ramp.end[2]:
        return (ramp.from_floor, ramp.start[:2]), (ramp.to_floor, ramp.end[:2])
    return (ramp.to_floor, ramp.end[:2]), (ramp.from_floor, ramp.start[:2])


def get_ramp_by_name(floorplan: Floorplan, ramp_name: Optional[str]) -> Optional[Ramp]:
    if not ramp_name:
        return None
    for ramp in floorplan.ramps:
        if ramp.name == ramp_name:
            return ramp
    return None


def ramp_endpoint_for_floor(ramp: Ramp, floor_index: int) -> Optional[Vector]:
    if floor_index == ramp.from_floor:
        return ramp.start[:2].copy()
    if floor_index == ramp.to_floor:
        return ramp.end[:2].copy()
    return None


def can_enter_ramp_from_floor(ramp: Ramp, floor_index: int, point_xy: Vector) -> bool:
    if floor_index != ramp.upper_floor:
        return False

    endpoint = ramp_endpoint_for_floor(ramp, floor_index)
    if endpoint is None:
        return False
    _, along, lateral, _, _, length = ramp_local_coordinates(ramp, point_xy)
    half_width = 0.5 * ramp.width
    lateral_margin = max(0.15, 0.35 * ramp.width)
    entry_depth = min(length * 0.4, max(1.25, 0.9 * ramp.width))

    if floor_index == ramp.from_floor:
        in_along_band = -0.15 <= along <= entry_depth
    elif floor_index == ramp.to_floor:
        in_along_band = (length - entry_depth) <= along <= (length + 0.15)
    else:
        return False

    in_width_band = abs(lateral) <= half_width + lateral_margin
    return in_along_band and in_width_band and norm(point_xy - endpoint) <= max(0.9, ramp.width)


def ramp_local_coordinates(ramp: Ramp, point_xy: Vector) -> Tuple[Vector, float, float, Vector, Vector, float]:
    start_xy = ramp.start[:2]
    end_xy = ramp.end[:2]
    axis = end_xy - start_xy
    length = max(norm(axis), EPS)
    tangent = axis / length
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    rel = point_xy - start_xy
    along = float(np.dot(rel, tangent))
    lateral = float(np.dot(rel, normal))
    closest_xy = start_xy + clamp(along, 0.0, length) * tangent
    return closest_xy, along, lateral, tangent, normal, length


def point_near_ramp(ramp: Ramp, point_xy: Vector, margin: float = 0.0) -> bool:
    _, along, lateral, _, _, length = ramp_local_coordinates(ramp, point_xy)
    half_width = 0.5 * ramp.width
    return (-margin <= along <= length + margin) and (abs(lateral) <= half_width + margin)


def clamp_point_to_ramp(ramp: Ramp, point_xy: Vector, agent_radius: float) -> Vector:
    _, along, lateral, tangent, normal, length = ramp_local_coordinates(ramp, point_xy)
    half_width = max(0.05, 0.5 * ramp.width - 0.55 * agent_radius)
    along_clamped = clamp(along, 0.0, length)
    lateral_clamped = clamp(lateral, -half_width, half_width)
    return ramp.start[:2] + along_clamped * tangent + lateral_clamped * normal


def ramps_for_floor(floorplan: Floorplan, floor_index: int) -> List[Ramp]:
    return [
        ramp
        for ramp in floorplan.ramps
        if floor_index in {ramp.from_floor, ramp.to_floor}
    ]


def build_navigation_field(floorplan: Floorplan, scenario: Scenario) -> NavigationField:
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("scipy is required to build the navigation field.") from exc

    all_polygons = [floor.polygon for floor in floorplan.floors] + [ramp.polygon for ramp in floorplan.ramps]
    xmins, xmaxs, ymins, ymaxs = zip(*(polygon_bbox(poly) for poly in all_polygons))
    padding = 1.0
    cell_size = scenario.simulation.cell_size
    x_coords = np.arange(min(xmins) - padding, max(xmaxs) + padding + cell_size, cell_size)
    y_coords = np.arange(min(ymins) - padding, max(ymaxs) + padding + cell_size, cell_size)
    floor_lookup = {floor.index: floor for floor in floorplan.floors}

    walkable_layers = []
    distance_layers = []
    for floor in floorplan.floors:
        mask = floor_walkable_mask(floor, floorplan, x_coords, y_coords)
        walkable_layers.append(mask)
        distance_layers.append(distance_transform_edt(mask) * cell_size)

    walkable = np.stack(walkable_layers, axis=0)
    distance_to_block = np.stack(distance_layers, axis=0)
    navigation = np.full_like(distance_to_block, INF, dtype=float)
    portal_edges: Dict[Tuple[int, int, int], List[PortalEdge]] = {}
    goal_floor = nearest_floor_index(floorplan.goal.center[2], floorplan.floors)

    xx, yy = np.meshgrid(x_coords, y_coords)
    goal_mask = (
        walkable[goal_floor]
        & (xx >= floorplan.goal.min_corner[0] - EPS)
        & (xx <= floorplan.goal.max_corner[0] + EPS)
        & (yy >= floorplan.goal.min_corner[1] - EPS)
        & (yy <= floorplan.goal.max_corner[1] + EPS)
    )
    if not np.any(goal_mask):
        nearest_goal = nearest_walkable_index(walkable[goal_floor], x_coords, y_coords, floorplan.goal.center[:2])
        if nearest_goal is None:
            raise RuntimeError("No walkable cells exist on the goal floor.")
        goal_mask[nearest_goal[0], nearest_goal[1]] = True

    for ramp in floorplan.ramps:
        (upper_floor, upper_xy), (lower_floor, lower_xy) = get_ramp_entry_exit(ramp)
        upper_nodes = []
        lower_nodes = []
        upper_mask = walkable[upper_floor]
        lower_mask = walkable[lower_floor]
        upper_idx = np.argwhere(upper_mask)
        lower_idx = np.argwhere(lower_mask)
        if len(upper_idx) == 0 or len(lower_idx) == 0:
            continue

        upper_xy_all = np.column_stack((x_coords[upper_idx[:, 1]], y_coords[upper_idx[:, 0]]))
        lower_xy_all = np.column_stack((x_coords[lower_idx[:, 1]], y_coords[lower_idx[:, 0]]))
        upper_dist = np.linalg.norm(upper_xy_all - upper_xy[None, :], axis=1)
        lower_dist = np.linalg.norm(lower_xy_all - lower_xy[None, :], axis=1)
        upper_keep = upper_idx[upper_dist <= max(ramp.width, 1.25)]
        lower_keep = lower_idx[lower_dist <= max(ramp.width, 1.25)]
        if len(upper_keep) == 0:
            best = upper_idx[int(np.argmin(upper_dist))]
            upper_keep = np.array([best])
        if len(lower_keep) == 0:
            best = lower_idx[int(np.argmin(lower_dist))]
            lower_keep = np.array([best])

        for uy, ux in upper_keep:
            upper_nodes.append((int(uy), int(ux)))
        for ly, lx in lower_keep:
            lower_nodes.append((int(ly), int(lx)))

        ramp_cost = max(norm(ramp.end - ramp.start), cell_size)
        for uy, ux in upper_nodes:
            source_key = (upper_floor, uy, ux)
            portal_edges.setdefault(source_key, [])
            source_xy = np.array([x_coords[ux], y_coords[uy]], dtype=float)
            for ly, lx in lower_nodes:
                target_xy = np.array([x_coords[lx], y_coords[ly]], dtype=float)
                cost = ramp_cost + 0.3 * (norm(source_xy - upper_xy) + norm(target_xy - lower_xy))
                portal_edges[source_key].append(PortalEdge(lower_floor, ly, lx, cost))
        for ly, lx in lower_nodes:
            source_key = (lower_floor, ly, lx)
            portal_edges.setdefault(source_key, [])
            source_xy = np.array([x_coords[lx], y_coords[ly]], dtype=float)
            for uy, ux in upper_nodes:
                target_xy = np.array([x_coords[ux], y_coords[uy]], dtype=float)
                cost = ramp_cost + 0.3 * (norm(source_xy - lower_xy) + norm(target_xy - upper_xy))
                portal_edges[source_key].append(PortalEdge(upper_floor, uy, ux, cost))

    neighbors = [
        (-1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
    ]
    pq: List[Tuple[float, Tuple[int, int, int]]] = []
    for iy, ix in np.argwhere(goal_mask):
        navigation[goal_floor, iy, ix] = 0.0
        heappush(pq, (0.0, (goal_floor, int(iy), int(ix))))

    ny = len(y_coords)
    nx = len(x_coords)
    while pq:
        current_dist, (floor_idx, iy, ix) = heappop(pq)
        if current_dist > navigation[floor_idx, iy, ix] + EPS:
            continue

        for dy, dx, w in neighbors:
            jy = iy + dy
            jx = ix + dx
            if jy < 0 or jy >= ny or jx < 0 or jx >= nx:
                continue
            if not walkable[floor_idx, jy, jx]:
                continue
            candidate = current_dist + w * cell_size
            if candidate + EPS < navigation[floor_idx, jy, jx]:
                navigation[floor_idx, jy, jx] = candidate
                heappush(pq, (candidate, (floor_idx, jy, jx)))

        for edge in portal_edges.get((floor_idx, iy, ix), []):
            if not walkable[edge.target_floor, edge.target_y, edge.target_x]:
                continue
            candidate = current_dist + edge.cost
            if candidate + EPS < navigation[edge.target_floor, edge.target_y, edge.target_x]:
                navigation[edge.target_floor, edge.target_y, edge.target_x] = candidate
                heappush(pq, (candidate, (edge.target_floor, edge.target_y, edge.target_x)))

    return NavigationField(
        cell_size=cell_size,
        x_coords=x_coords,
        y_coords=y_coords,
        walkable=walkable,
        distance_to_block=distance_to_block,
        navigation=navigation,
        portal_edges=portal_edges,
        goal_floor=goal_floor,
        floor_lookup=floor_lookup,
    )


def relative_sample_on_floor(floor: Floor, placement: Dict) -> Vector:
    bx0, bx1, by0, by1 = floor.bbox
    center = np.array([0.5 * (bx0 + bx1), 0.5 * (by0 + by1)], dtype=float)
    extent = np.array([bx1 - bx0, by1 - by0], dtype=float)

    if "absolute_box" in placement:
        box = placement["absolute_box"]
        min_corner = vec(box["min"])
        max_corner = vec(box["max"])
        return np.array(
            [
                random.uniform(min_corner[0], max_corner[0]),
                random.uniform(min_corner[1], max_corner[1]),
            ],
            dtype=float,
        )

    xr = placement.get("offset_range", {}).get("x")
    yr = placement.get("offset_range", {}).get("y")
    if xr is None or yr is None:
        base = placement.get("offset", [0.0, 0.0])
        jitter = placement.get("jitter", [0.08, 0.08])
        xr = [normalize_percent(base[0]) - abs(normalize_percent(jitter[0])), normalize_percent(base[0]) + abs(normalize_percent(jitter[0]))]
        yr = [normalize_percent(base[1]) - abs(normalize_percent(jitter[1])), normalize_percent(base[1]) + abs(normalize_percent(jitter[1]))]
    xr = [normalize_percent(float(xr[0])), normalize_percent(float(xr[1]))]
    yr = [normalize_percent(float(yr[0])), normalize_percent(float(yr[1]))]

    for _ in range(300):
        rel = np.array([
            random.uniform(xr[0], xr[1]),
            random.uniform(yr[0], yr[1]),
        ], dtype=float)
        point_xy = center + 0.5 * extent * rel
        if point_in_polygon(point_xy, floor.polygon):
            return point_xy
    return center.copy()


def support_surface_z(floorplan: Floorplan, floor_index: int, point_xy: Vector) -> Tuple[float, Optional[Ramp]]:
    candidate_ramp = None
    for ramp in ramps_for_floor(floorplan, floor_index):
        if point_in_polygon(point_xy, ramp.polygon) and can_enter_ramp_from_floor(
            ramp,
            floor_index,
            point_xy,
        ):
            candidate_ramp = ramp
            break
    if candidate_ramp is not None:
        return ramp_height_at_xy(candidate_ramp, point_xy), candidate_ramp
    floor = floorplan.floors[floor_index]
    return floor.surface_z, None


def resolve_spawn_position(field: NavigationField, floorplan: Floorplan, floor_index: int, point_xy: Vector, radius: float) -> Tuple[Vector, int, Optional[str]]:
    snapped_xy = nearest_walkable_xy(field, floor_index, point_xy)
    z_value, ramp = support_surface_z(floorplan, floor_index, snapped_xy)
    return np.array([snapped_xy[0], snapped_xy[1], z_value + radius], dtype=float), floor_index, None if ramp is None else ramp.name


def initialize_agents(scenario: Scenario, floorplan: Floorplan, field: NavigationField) -> List[Agent]:
    agents: List[Agent] = []
    for group in scenario.groups:
        floor = floorplan.floors[group.floor]
        for _ in range(group.count):
            point_xy = relative_sample_on_floor(floor, group.placement)
            position, floor_index, ramp_name = resolve_spawn_position(
                field,
                floorplan,
                group.floor,
                point_xy,
                scenario.agent_defaults.radius,
            )
            agents.append(
                Agent(
                    pos=position,
                    vel=np.zeros(3, dtype=float),
                    floor_index=floor_index,
                    radius=scenario.agent_defaults.radius,
                    color=group.color,
                    repulsion=group.repulsion,
                    ramp_name=ramp_name,
                )
            )
    return agents


def anisotropic_weight(forward_dir: Vector, direction_to_other: Vector, cfg: RepulsionConfig) -> float:
    if not cfg.anisotropy_enabled:
        return 1.0
    fd = unit(forward_dir)
    od = unit(direction_to_other)
    if norm(fd) < EPS or norm(od) < EPS:
        return 1.0
    angle = math.degrees(math.acos(clamp(float(np.dot(fd, od)), -1.0, 1.0)))
    return cfg.forward_strength if angle <= cfg.forward_angle_deg else cfg.backward_strength


def choose_guiding_ramp(agent: Agent, floorplan: Floorplan, field: NavigationField) -> Optional[Ramp]:
    best_ramp = None
    best_value = INF
    agent_xy = agent.pos[:2]
    for ramp in floorplan.ramps:
        (upper_floor, upper_xy), _ = get_ramp_entry_exit(ramp)
        if agent.floor_index != upper_floor:
            continue
        nav_value = sample_scalar(field.navigation[upper_floor], field.x_coords, field.y_coords, upper_xy)
        if nav_value >= INF * 0.5:
            continue
        score = nav_value + 0.75 * norm(agent_xy - upper_xy)
        if score < best_value:
            best_value = score
            best_ramp = ramp
    return best_ramp


def compute_navigation_force(agent: Agent, floorplan: Floorplan, field: NavigationField, cfg: AgentConfig) -> Vector:
    agent_xy = agent.pos[:2]

    active_ramp = get_ramp_by_name(floorplan, agent.ramp_name)
    if active_ramp is not None:
        (_, upper_xy), (lower_floor, lower_xy) = get_ramp_entry_exit(active_ramp)
        centerline_xy, along, lateral, tangent, normal, length = ramp_local_coordinates(
            active_ramp,
            agent_xy,
        )
        downhill = tangent if active_ramp.end[2] < active_ramp.start[2] else -tangent
        if lower_floor != active_ramp.lower_floor:
            downhill = -downhill

        center_correction = centerline_xy - agent_xy
        exit_target = lower_xy - agent_xy
        progress = clamp(along / max(length, EPS), 0.0, 1.0)
        half_width = max(0.5 * active_ramp.width, EPS)
        lateral_ratio = clamp(abs(lateral) / half_width, 0.0, 1.0)

        force_xy = 1.35 * cfg.ramp_gain * downhill
        force_xy += 0.9 * cfg.ramp_gain * lateral_ratio * unit(center_correction)
        force_xy += (0.55 + 0.35 * progress) * cfg.ramp_gain * unit(exit_target)

        if norm(exit_target) < max(0.8, 0.6 * active_ramp.width):
            nav_grid = field.navigation[active_ramp.lower_floor]
            grad = sample_gradient(nav_grid, field.x_coords, field.y_coords, agent_xy, field.cell_size)
            if np.all(np.isfinite(grad)) and norm(grad) > 1e-6:
                force_xy += 0.4 * cfg.nav_gain * (-unit(grad))

        return np.array([force_xy[0], force_xy[1], 0.0], dtype=float)

    nav_grid = field.navigation[agent.floor_index]
    grad = sample_gradient(nav_grid, field.x_coords, field.y_coords, agent_xy, field.cell_size)
    if not np.all(np.isfinite(grad)) or norm(grad) < 1e-6:
        target = floorplan.goal.center[:2]
        return cfg.nav_gain * unit(target - agent_xy)

    force_xy = -cfg.nav_gain * unit(grad)
    if agent.floor_index == field.goal_floor:
        goal_pull = unit(floorplan.goal.center[:2] - agent_xy)
        force_xy += cfg.goal_gain * goal_pull

    ramp = choose_guiding_ramp(agent, floorplan, field)
    if ramp is not None:
        (upper_floor, upper_xy), (lower_floor, lower_xy) = get_ramp_entry_exit(ramp)
        if agent.floor_index == upper_floor:
            to_portal = lower_xy - upper_xy
            tangent = ramp_tangent_xy(ramp, toward_lower=True)
            dist_to_polygon = 0.0 if point_in_polygon(agent_xy, ramp.polygon) else norm(agent_xy - upper_xy)
            entry_target = upper_xy - agent_xy
            if can_enter_ramp_from_floor(ramp, agent.floor_index, agent_xy):
                centerline_at, _ = closest_point_on_segment_2d(agent_xy, ramp.start[:2], ramp.end[:2])
                align = unit(centerline_at - agent_xy)
                force_xy += cfg.ramp_gain * tangent + 0.6 * cfg.ramp_gain * align
            elif dist_to_polygon < max(2.0 * ramp.width, 2.5):
                centerline_at, _ = closest_point_on_segment_2d(agent_xy, ramp.start[:2], ramp.end[:2])
                align = unit(centerline_at - agent_xy)
                force_xy += 1.15 * cfg.ramp_gain * unit(entry_target)
                force_xy += 0.35 * cfg.ramp_gain * align
            elif norm(to_portal) > EPS:
                force_xy += 0.45 * cfg.ramp_gain * unit(upper_xy - agent_xy)
    return np.array([force_xy[0], force_xy[1], 0.0], dtype=float)


def compute_wall_force(agent: Agent, field: NavigationField, cfg: AgentConfig) -> Vector:
    dist_grid = field.distance_to_block[agent.floor_index]
    d = sample_scalar(dist_grid, field.x_coords, field.y_coords, agent.pos[:2])
    effective_range = min(cfg.wall_range, max(0.45, 2.4 * agent.radius + 0.12))
    if not math.isfinite(d) or d <= EPS or d >= effective_range:
        return np.zeros(3, dtype=float)
    grad_d = sample_gradient(dist_grid, field.x_coords, field.y_coords, agent.pos[:2], field.cell_size)
    direction = unit(grad_d)
    magnitude = cfg.wall_strength * ((1.0 / max(d, 0.05)) - (1.0 / effective_range)) / max(d * d, 0.05)
    return np.array([direction[0], direction[1], 0.0], dtype=float) * magnitude


def compute_ramp_edge_force(agent: Agent, floorplan: Floorplan, cfg: AgentConfig) -> Vector:
    ramp = get_ramp_by_name(floorplan, agent.ramp_name)
    if ramp is None:
        return np.zeros(3, dtype=float)

    _, along, lateral, _, normal, length = ramp_local_coordinates(ramp, agent.pos[:2])
    if along < -agent.radius or along > length + agent.radius:
        return np.zeros(3, dtype=float)

    half_width = 0.5 * ramp.width
    distance_to_edge = half_width - abs(lateral)
    if distance_to_edge >= cfg.wall_range:
        return np.zeros(3, dtype=float)

    sign = -1.0 if lateral >= 0.0 else 1.0
    direction = sign * normal
    clearance = max(distance_to_edge, 0.03)
    magnitude = cfg.wall_strength * ((1.0 / clearance) - (1.0 / cfg.wall_range)) / max(clearance * clearance, 0.03)
    if distance_to_edge < 0.0:
        magnitude *= 2.5
    return np.array([direction[0], direction[1], 0.0], dtype=float) * magnitude


def compute_social_force(index: int, agents: Sequence[Agent], floorplan: Floorplan, cfg: AgentConfig) -> Vector:
    agent = agents[index]
    total = np.zeros(3, dtype=float)
    for j, other in enumerate(agents):
        if j == index or not other.active or other.reached_goal:
            continue
        delta = agent.pos - other.pos
        distance = norm(delta)
        if distance < EPS or distance > max(cfg.social_range, other.repulsion.range):
            continue
        weight = anisotropic_weight(other.vel, agent.pos - other.pos, other.repulsion)
        effective_range = max(0.25, other.repulsion.range)
        if distance >= effective_range:
            continue
        direction = unit(delta)
        strength = cfg.social_strength * other.repulsion.strength * weight
        total += direction * strength * ((1.0 / max(distance, 0.1)) - (1.0 / effective_range)) / max(distance, 0.15)
    if distance_point_to_aabb(agent.pos, floorplan.goal.min_corner, floorplan.goal.max_corner) <= 2.0 * agent.radius:
        total *= 0.1
    return total


def compute_agent_force(index: int, agents: Sequence[Agent], floorplan: Floorplan, field: NavigationField, cfg: AgentConfig) -> Vector:
    agent = agents[index]
    if not agent.active:
        return np.zeros(3, dtype=float)
    force = np.zeros(3, dtype=float)
    navigation_force = compute_navigation_force(agent, floorplan, field, cfg)
    wall_force = compute_wall_force(agent, field, cfg)
    ramp_edge_force = compute_ramp_edge_force(agent, floorplan, cfg)

    nav_norm = norm(navigation_force[:2])
    if nav_norm > EPS:
        nav_dir = navigation_force[:2] / nav_norm
        wall_against_nav = float(np.dot(wall_force[:2], nav_dir))
        if wall_against_nav < 0.0:
            wall_force[:2] -= wall_against_nav * nav_dir

    force += navigation_force
    force += wall_force
    force += ramp_edge_force
    force += compute_social_force(index, agents, floorplan, cfg)
    return force


def project_agent_to_walkable(agent: Agent, floorplan: Floorplan, field: NavigationField) -> None:
    current_ramp = get_ramp_by_name(floorplan, agent.ramp_name)
    if current_ramp is not None and point_near_ramp(
        current_ramp,
        agent.pos[:2],
        margin=max(agent.radius * 2.0, 0.2),
    ):
        agent.pos[:2] = clamp_point_to_ramp(current_ramp, agent.pos[:2], agent.radius)
        return

    if is_xy_walkable_on_floor(floorplan, agent.floor_index, agent.pos[:2]):
        return
    snapped_xy = nearest_walkable_xy(field, agent.floor_index, agent.pos[:2])
    agent.pos[:2] = snapped_xy
    agent.vel[:2] *= 0.25


def update_agent_surface(agent: Agent, floorplan: Floorplan, field: NavigationField) -> None:
    xy = agent.pos[:2]
    ramp = get_ramp_by_name(floorplan, agent.ramp_name)
    if ramp is not None and not point_near_ramp(ramp, xy, margin=max(agent.radius * 2.0, 0.2)):
        ramp = None

    if ramp is None:
        for candidate in ramps_for_floor(floorplan, agent.floor_index):
            if not can_enter_ramp_from_floor(candidate, agent.floor_index, xy):
                continue
            if point_in_polygon(xy, candidate.polygon) or point_near_ramp(
                candidate,
                xy,
                margin=max(agent.radius * 1.5, 0.15),
            ):
                ramp = candidate
                break

    if ramp is not None:
        agent.pos[:2] = clamp_point_to_ramp(ramp, xy, agent.radius)
        xy = agent.pos[:2]
        agent.ramp_name = ramp.name
        high_floor = ramp.from_floor if ramp.start[2] >= ramp.end[2] else ramp.to_floor
        low_floor = ramp.to_floor if high_floor == ramp.from_floor else ramp.from_floor
        _, t = closest_point_on_segment_2d(xy, ramp.start[:2], ramp.end[:2])
        agent.floor_index = high_floor if t < 0.5 else low_floor
        agent.pos[2] = ramp_height_at_xy(ramp, xy) + agent.radius
        if norm(xy - get_ramp_entry_exit(ramp)[1][1]) <= max(ramp.width * 0.4, 0.55):
            agent.floor_index = get_ramp_entry_exit(ramp)[1][0]
            agent.ramp_name = None
        return

    floor = floorplan.floors[agent.floor_index]
    if not point_in_polygon(xy, floor.polygon):
        best_floor = agent.floor_index
        best_distance = INF
        for candidate in floorplan.floors:
            snapped = nearest_walkable_xy(field, candidate.index, xy)
            dist = norm(snapped - xy)
            if dist < best_distance:
                best_distance = dist
                best_floor = candidate.index
        agent.floor_index = best_floor
        floor = floorplan.floors[best_floor]
    agent.ramp_name = None
    agent.pos[2] = floor.surface_z + agent.radius


def step_simulation(agents: List[Agent], floorplan: Floorplan, field: NavigationField, cfg: AgentConfig, dt: float) -> None:
    forces = [compute_agent_force(i, agents, floorplan, field, cfg) for i in range(len(agents))]
    for agent, force in zip(agents, forces):
        if not agent.active:
            continue

        desired_velocity = force
        agent.vel = (1.0 - cfg.damping) * agent.vel + cfg.damping * desired_velocity
        speed = norm(agent.vel)
        if speed > cfg.max_speed:
            agent.vel *= cfg.max_speed / speed

        agent.pos[:2] += agent.vel[:2] * dt
        agent.pos[0] = clamp(agent.pos[0], floorplan.bounds_min[0], floorplan.bounds_max[0])
        agent.pos[1] = clamp(agent.pos[1], floorplan.bounds_min[1], floorplan.bounds_max[1])
        project_agent_to_walkable(agent, floorplan, field)
        update_agent_surface(agent, floorplan, field)

        if distance_point_to_aabb(agent.pos, floorplan.goal.min_corner, floorplan.goal.max_corner) <= agent.radius:
            agent.reached_goal = True
            agent.active = False
            agent.vel[:] = 0.0
            goal_center = floorplan.goal.center.copy()
            goal_center[2] = max(goal_center[2], floorplan.goal.min_corner[2] + agent.radius)
            agent.pos[:] = goal_center


def create_mesh_actor(vertices: List[List[float]], faces: List[List[int]], color: str, alpha: float):
    from vedo import Mesh

    mesh = Mesh([vertices, faces])
    mesh.c(color).alpha(alpha)
    return mesh


def build_scene(floorplan: Floorplan):
    from vedo import Box, Cylinder

    actors = []

    for floor in floorplan.floors:
        verts, faces = build_surface_mesh(floor.polygon, floor.surface_z)
        actors.append(create_mesh_actor(verts, faces, floor.color, 0.12))

    for fixture in floorplan.fixtures:
        if fixture.kind == "column":
            assert fixture.center is not None and fixture.radius is not None
            height = fixture.top_z - fixture.base_z
            cyl = Cylinder(
                pos=(fixture.center[0], fixture.center[1], fixture.base_z + 0.5 * height),
                r=fixture.radius,
                height=height,
                axis=(0, 0, 1),
                c=fixture.color,
                alpha=fixture.alpha,
            )
            actors.append(cyl)
        elif fixture.polygon is not None:
            verts, faces = build_prism_mesh(fixture.polygon, fixture.base_z, fixture.top_z)
            actors.append(create_mesh_actor(verts, faces, fixture.color, fixture.alpha))

    for floor in floorplan.floors:
        for obstacle in floor.obstacles:
            if obstacle.kind == "column":
                assert obstacle.center is not None and obstacle.radius is not None
                height = obstacle.top_z - obstacle.base_z
                cyl = Cylinder(
                    pos=(obstacle.center[0], obstacle.center[1], obstacle.base_z + 0.5 * height),
                    r=obstacle.radius,
                    height=height,
                    axis=(0, 0, 1),
                    c=obstacle.color,
                    alpha=obstacle.alpha,
                )
                actors.append(cyl)
            elif obstacle.polygon is not None:
                verts, faces = build_prism_mesh(obstacle.polygon, obstacle.base_z, obstacle.top_z)
                actors.append(create_mesh_actor(verts, faces, obstacle.color, obstacle.alpha))

    for ramp in floorplan.ramps:
        verts, faces = build_ramp_mesh(ramp)
        actors.append(create_mesh_actor(verts, faces, ramp.color, 0.6))

    goal_size = floorplan.goal.max_corner - floorplan.goal.min_corner
    goal_actor = Box(
        pos=floorplan.goal.center,
        length=goal_size[0],
        width=goal_size[1],
        height=goal_size[2],
    )
    goal_actor.c(floorplan.goal.color).alpha(0.45)
    actors.append(goal_actor)
    return actors


def run_simulation_without_render(
    floorplan: Floorplan,
    scenario: Scenario,
    agents: List[Agent],
    field: NavigationField,
) -> None:
    for _ in range(scenario.simulation.steps):
        step_simulation(
            agents,
            floorplan,
            field,
            scenario.agent_defaults,
            scenario.simulation.dt,
        )

    active = sum(1 for agent in agents if agent.active)
    reached = sum(1 for agent in agents if agent.reached_goal)
    print(
        f"Headless run complete: steps={scenario.simulation.steps} active={active} reached={reached}"
    )


def render_simulation(
    floorplan: Floorplan,
    scenario: Scenario,
    agents: List[Agent],
    field: NavigationField,
    output_video: Optional[str],
    offscreen: bool,
) -> None:
    if offscreen and output_video is None:
        run_simulation_without_render(floorplan, scenario, agents, field)
        return

    try:
        from vedo import Plotter, Sphere, Text2D, Video
    except ImportError as exc:
        raise RuntimeError("vedo is required for rendering. Install project requirements first.") from exc

    plotter = Plotter(title=floorplan.name, size=(1280, 900))
    plotter.show(*build_scene(floorplan), interactive=False, resetcam=True)

    center = 0.5 * (floorplan.bounds_min + floorplan.bounds_max)
    scene_size = floorplan.bounds_max - floorplan.bounds_min
    max_dim = float(np.max(scene_size))
    cam_pos = center + np.array([1.25, 1.2, 1.0], dtype=float) * max_dim
    plotter.camera.SetPosition(cam_pos.tolist())
    plotter.camera.SetFocalPoint(center.tolist())
    plotter.camera.SetViewUp((0, 0, 1))
    plotter.camera.SetParallelProjection(True)
    plotter.camera.SetParallelScale(0.58 * max(scene_size[0], scene_size[1]))
    plotter.reset_clipping_range()
    plotter.render()

    sphere_actors = []
    for agent in agents:
        actor = Sphere(pos=agent.pos, r=agent.radius, c=agent.color)
        actor.alpha(0.95)
        sphere_actors.append(actor)
    if sphere_actors:
        plotter.add(*sphere_actors)

    status = Text2D("", pos="top-left", c="white", font="Courier", s=0.8)
    plotter.add(status)

    writer = None
    if output_video:
        writer = Video(output_video, fps=max(1, int(round(1.0 / scenario.simulation.dt))))

    for step in range(scenario.simulation.steps):
        step_simulation(agents, floorplan, field, scenario.agent_defaults, scenario.simulation.dt)
        for actor, agent in zip(sphere_actors, agents):
            actor.pos(agent.pos)
            actor.alpha(0.35 if agent.reached_goal else 0.95)
        active = sum(1 for agent in agents if agent.active)
        reached = sum(1 for agent in agents if agent.reached_goal)
        status.text(f"step {step + 1}/{scenario.simulation.steps}   active: {active}   reached: {reached}")
        plotter.render()
        if writer is not None:
            writer.add_frame()

    if writer is not None:
        writer.close()

    if offscreen:
        plotter.close()
    else:
        plotter.interactive().close()


def make_example_data(root: pathlib.Path) -> None:
    floorplan_dir = root / "data" / "floorplans"
    scenario_dir = root / "data" / "scenarios"
    floorplan_dir.mkdir(parents=True, exist_ok=True)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    floorplan = {
        "name": "example_three_story",
        "grid": {"cell_size": 0.35, "padding": 1.0},
        "floors": [
            {
                "name": "ground",
                "z": 0.0,
                "thickness": 0.18,
                "polygon": [[0, 0], [18, 0], [18, 12], [0, 12]],
                "obstacles": [
                    {"type": "box", "polygon": [[5.2, 4.1], [6.7, 4.1], [6.7, 5.8], [5.2, 5.8]], "height": 1.25, "color": "brown4"},
                    {"type": "column", "center": [12.5, 8.3], "radius": 0.45, "height": 3.0, "color": "tan"},
                ],
            },
            {
                "name": "mid",
                "z": 3.0,
                "thickness": 0.18,
                "polygon": [[0, 0], [18, 0], [18, 12], [0, 12]],
                "obstacles": [
                    {"type": "box", "polygon": [[9.0, 6.1], [10.8, 6.1], [10.8, 7.6], [9.0, 7.6]], "height": 1.2, "color": "brown4"},
                ],
            },
            {
                "name": "top",
                "z": 6.0,
                "thickness": 0.18,
                "polygon": [[0, 0], [18, 0], [18, 12], [0, 12]],
                "obstacles": [
                    {"type": "column", "center": [6.4, 8.4], "radius": 0.5, "height": 2.8, "color": "tan"},
                ],
            },
        ],
        "fixtures": [
            {"type": "wall", "polygon": [[0, 0], [18, 0], [18, 0.22], [0, 0.22]], "z": [0, 9.4], "color": "slategray"},
            {"type": "wall", "polygon": [[0, 11.78], [18, 11.78], [18, 12], [0, 12]], "z": [0, 9.4], "color": "slategray"},
            {"type": "wall", "polygon": [[0, 0], [0.22, 0], [0.22, 12], [0, 12]], "z": [0, 9.4], "color": "slategray"},
            {"type": "wall", "polygon": [[17.78, 0], [18, 0], [18, 12], [17.78, 12]], "z": [0, 9.4], "color": "slategray"},
        ],
        "ramps": [
            {
                "name": "top_to_mid",
                "start": [17.0, 8.1, 6.09],
                "end": [12.5, 8.1, 3.09],
                "width": 2.0,
                "thickness": 0.16,
                "from_floor": 2,
                "to_floor": 1,
            },
            {
                "name": "mid_to_ground",
                "start": [5.0, 3.8, 3.09],
                "end": [9.8, 3.8, 0.09],
                "width": 2.0,
                "thickness": 0.16,
                "from_floor": 1,
                "to_floor": 0,
            },
        ],
        "openings": [
            {"min": [12.2, 7.0, 5.72], "max": [17.4, 9.2, 6.45]},
            {"min": [4.6, 2.8, 2.72], "max": [10.2, 4.8, 3.45]},
            {"min": [-0.5, 5.0, -0.2], "max": [0.6, 7.0, 1.2]},
        ],
        "goal": {"min": [-0.15, 5.2, -0.05], "max": [0.7, 6.8, 1.0], "color": "green4"},
    }

    scenario = {
        "name": "example_mixed_crowd",
        "simulation": {"dt": 0.05, "steps": 550, "seed": 7},
        "agent_defaults": {
            "radius": 0.22,
            "max_speed": 1.35,
            "nav_gain": 3.0,
            "wall_strength": 5.4,
            "wall_range": 1.0,
            "social_strength": 1.9,
            "social_range": 1.1,
            "ramp_gain": 3.8,
            "goal_gain": 2.4,
            "damping": 0.72,
        },
        "agents": [
            {
                "count": 24,
                "floor": 2,
                "offset_range": {"x": [-0.75, 0.7], "y": [-0.6, 0.65]},
                "color": "royalblue",
                "repulsion": {
                    "strength": 1.9,
                    "range": 1.05,
                    "anisotropy": {
                        "enabled": True,
                        "forward_strength": 1.7,
                        "backward_strength": 0.55,
                        "forward_angle_deg": 80,
                    },
                },
            },
            {
                "count": 18,
                "floor": 1,
                "offset_range": {"x": [-0.7, 0.75], "y": [-0.65, 0.55]},
                "color": "tomato",
                "repulsion": {
                    "strength": 1.6,
                    "range": 0.95,
                    "anisotropy": {"enabled": False},
                },
            },
            {
                "count": 14,
                "floor": 0,
                "offset_range": {"x": [-0.75, 0.75], "y": [-0.55, 0.55]},
                "color": "goldenrod",
                "repulsion": {
                    "strength": 1.4,
                    "range": 0.9,
                    "anisotropy": {
                        "enabled": True,
                        "forward_strength": 1.45,
                        "backward_strength": 0.7,
                        "forward_angle_deg": 70,
                    },
                },
            },
        ],
        "obstacles": [
            {"type": "box", "floor": 1, "offset": [0.18, -0.24], "size": [1.0, 1.3], "height": 1.15, "color": "sienna"},
            {"type": "column", "floor": 0, "offset": [0.3, 0.15], "radius": 0.38, "height": 2.8, "color": "tan"},
        ],
    }

    with open(floorplan_dir / "example.json", "w", encoding="utf-8") as handle:
        json.dump(floorplan, handle, indent=2)
    with open(scenario_dir / "example.json", "w", encoding="utf-8") as handle:
        json.dump(scenario, handle, indent=2)
    print(f"Wrote example floorplan to {(floorplan_dir / 'example.json').as_posix()}")
    print(f"Wrote example scenario to {(scenario_dir / 'example.json').as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient-descent evacuation simulation with vedo visualization.")
    parser.add_argument("--floorplan", type=str, default=None, help="Floorplan file stem or path")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario file stem or path")
    parser.add_argument("--steps", type=int, default=None, help="Override simulation step count")
    parser.add_argument("--dt", type=float, default=None, help="Override time step")
    parser.add_argument("--video", type=str, default=None, help="Output video path")
    parser.add_argument("--offscreen", action="store_true", help="Render without opening a window")
    parser.add_argument("--make-example-data", action="store_true", help="Write example floorplan and scenario JSON files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parent
    if args.make_example_data:
        make_example_data(root)
        return

    if not args.floorplan or not args.scenario:
        print("Please provide --floorplan and --scenario, or use --make-example-data.")
        return

    floorplan = load_floorplan(args.floorplan)
    scenario = load_scenario(args.scenario, floorplan)
    apply_scenario_obstacles(floorplan, scenario)

    if args.steps is not None:
        scenario.simulation.steps = int(args.steps)
    if args.dt is not None:
        scenario.simulation.dt = float(args.dt)

    headless = not os.environ.get("DISPLAY")
    if args.offscreen:
        scenario.simulation.offscreen = True
    if headless:
        if not scenario.simulation.offscreen:
            print("DISPLAY is not set; using offscreen rendering.")
        scenario.simulation.offscreen = True

    random.seed(scenario.simulation.seed)
    np.random.seed(scenario.simulation.seed)

    field = build_navigation_field(floorplan, scenario)
    agents = initialize_agents(scenario, floorplan, field)

    render_simulation(
        floorplan=floorplan,
        scenario=scenario,
        agents=agents,
        field=field,
        output_video=args.video,
        offscreen=scenario.simulation.offscreen,
    )


if __name__ == "__main__":
    main()
