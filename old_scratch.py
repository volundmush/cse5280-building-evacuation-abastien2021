"""
Gradient-descent evacuation simulation with JSON floorplans/scenarios and vedo animation output.

Usage examples:
    python scratch.py --floorplan data/floorplans/example.json --scenario data/scenarios/example.json --steps 1200 --dt 0.05
    python scratch.py --make-example-data

Data format (JSON)
------------------
Floorplan (data/floorplans/*.json):
{
  "name": "Example 3-floor",
  "bounds": {"x": [0, 20], "y": [0, 12], "z": [0, 9]},
  "floors": [
    {"z": 0, "thickness": 0.2, "walkable": true},
    {"z": 3, "thickness": 0.2, "walkable": true},
    {"z": 6, "thickness": 0.2, "walkable": true}
  ],
  "walls": [
    {"min": [0,0,0], "max": [20,0.3,9]},
    {"min": [0,11.7,0], "max": [20,12,9]},
    {"min": [0,0,0], "max": [0.3,12,9]},
    {"min": [19.7,0,0], "max": [20,12,9]}
  ],
  "obstacles": [
    {"type": "box", "min": [6,4,0], "max": [8,6,3]},
    {"type": "pillar", "center": [14,8,0], "radius": 0.6, "height": 9}
  ],
  "stairs": [
    {"start": [10,5,6.1], "end": [10,7,3.1], "width": 2.0},
    {"start": [10,5,3.1], "end": [10,7,0.1], "width": 2.0}
  ],
  "goal": {"center": [10,0.8,0], "radius": 0.7}
}

Scenario (data/scenarios/*.json):
{
  "name": "Mixed crowd",
  "evacuees": [
    {
      "count": 35,
      "spawn": {
        "box": {"min": [2,2,0.2], "max": [18,10,2.8]}
      },
      "repulsion": {
        "strength": 2.5,
        "range": 2.2,
        "anisotropy": {
          "enabled": true,
          "forward_strength": 1.5,
          "backward_strength": 0.5,
          "forward_angle_deg": 70
        }
      }
    },
    {
      "count": 15,
      "spawn": {
        "box": {"min": [2,2,3.2], "max": [18,10,5.8]}
      },
      "repulsion": {
        "strength": 2.0,
        "range": 2.0,
        "anisotropy": {"enabled": false}
      }
    }
  ],
  "agent": {
    "radius": 0.25,
    "max_speed": 1.6,
    "goal_attraction": 3.5,
    "wall_repulsion": 6.0,
    "wall_range": 1.5,
    "floor_repulsion": 6.0,
    "floor_range": 0.7,
    "ramp_attraction": 2.5,
    "ramp_range": 4.0,
    "damping": 0.7
  },
  "simulation": {
    "dt": 0.05,
    "steps": 1200,
    "seed": 1337
  }
}

Notes:
- Gradient descent is computed from a scalar potential field composed of:
  * Attractive pull to goal
  * Repulsion from walls/obstacles
  * Repulsion from other agents (anisotropic or isotropic)
  * Floors act as walkable planes; agents stay on their current floor
- Once an evacuee reaches the goal, their repulsion field is disabled.
- Vedo animation is rendered for visualization.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from vedo import (
    Box,
    Cylinder,
    Plotter,
    Points,
    Sphere,
    Text2D,
    Video,
)

Vector = np.ndarray

data_folder = pathlib.Path() / "data"

# ----------------------------- Data Structures -----------------------------


@dataclass
class Floor:
    z: float
    thickness: float = 0.2
    walkable: bool = True


@dataclass
class Wall:
    min: Vector
    max: Vector


@dataclass
class Obstacle:
    kind: str
    min: Optional[Vector] = None
    max: Optional[Vector] = None
    center: Optional[Vector] = None
    radius: Optional[float] = None
    height: Optional[float] = None


@dataclass
class RepulsionMask:
    min: Vector
    max: Vector


@dataclass
class Stair:
    start: Vector
    end: Vector
    width: float


@dataclass
class Goal:
    center: Vector
    radius: float


@dataclass
class Floorplan:
    name: str
    bounds: Tuple[Vector, Vector]
    floors: List[Floor]
    walls: List[Wall]
    obstacles: List[Obstacle]
    repulsion_masks: List[RepulsionMask]
    stairs: List[Stair]
    goal: Goal


@dataclass
class RepulsionConfig:
    strength: float = 2.5
    range: float = 2.0
    anisotropy_enabled: bool = False
    forward_strength: float = 1.0
    backward_strength: float = 1.0
    forward_angle_deg: float = 90.0


@dataclass
class AgentConfig:
    radius: float = 0.25
    max_speed: float = 1.6
    goal_attraction: float = 3.0
    wall_repulsion: float = 6.0
    wall_range: float = 1.5
    floor_repulsion: float = 6.0
    floor_range: float = 0.7
    ramp_attraction: float = 2.5
    ramp_range: float = 4.0
    damping: float = 0.7


@dataclass
class SimulationConfig:
    dt: float = 0.05
    steps: int = 1200
    seed: int = 1337


@dataclass
class ScenarioGroup:
    count: int
    spawn_min: Vector
    spawn_max: Vector
    repulsion: RepulsionConfig


@dataclass
class Scenario:
    name: str
    groups: List[ScenarioGroup]
    agent_cfg: AgentConfig
    sim_cfg: SimulationConfig


@dataclass
class Agent:
    pos: Vector
    vel: Vector
    radius: float
    repulsion: RepulsionConfig
    active: bool = True
    reached_goal: bool = False


# ----------------------------- JSON Loading -----------------------------


def _vec(v) -> Vector:
    return np.array(v, dtype=float)


def load_floorplan(path: str) -> Floorplan:
    folder = data_folder / "floorplans"
    with open(folder / f"{path}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    name = data.get("name", os.path.basename(path))
    bounds = data.get("bounds", {"x": [0, 10], "y": [0, 10], "z": [0, 3]})
    bounds_min = _vec([bounds["x"][0], bounds["y"][0], bounds["z"][0]])
    bounds_max = _vec([bounds["x"][1], bounds["y"][1], bounds["z"][1]])

    floors = []
    for fl in data.get("floors", []):
        floors.append(
            Floor(
                z=fl["z"],
                thickness=fl.get("thickness", 0.2),
                walkable=fl.get("walkable", True),
            )
        )

    walls = []
    for w in data.get("walls", []):
        walls.append(Wall(min=_vec(w["min"]), max=_vec(w["max"])))

    obstacles = []
    for ob in data.get("obstacles", []):
        kind = ob.get("type", "box")
        if kind == "box":
            obstacles.append(
                Obstacle(kind="box", min=_vec(ob["min"]), max=_vec(ob["max"]))
            )
        elif kind == "pillar":
            obstacles.append(
                Obstacle(
                    kind="pillar",
                    center=_vec(ob["center"]),
                    radius=float(ob["radius"]),
                    height=float(ob.get("height", bounds_max[2] - bounds_min[2])),
                )
            )

    masks = []
    for m in data.get("repulsion_masks", []):
        masks.append(RepulsionMask(min=_vec(m["min"]), max=_vec(m["max"])))

    stairs = []
    for st in data.get("stairs", []):
        stairs.append(
            Stair(
                start=_vec(st["start"]),
                end=_vec(st["end"]),
                width=float(st.get("width", 1.0)),
            )
        )

    g = data.get(
        "goal", {"center": [bounds_min[0], bounds_min[1], bounds_min[2]], "radius": 0.5}
    )
    goal = Goal(center=_vec(g["center"]), radius=float(g.get("radius", 0.5)))

    return Floorplan(
        name=name,
        bounds=(bounds_min, bounds_max),
        floors=floors,
        walls=walls,
        obstacles=obstacles,
        repulsion_masks=masks,
        stairs=stairs,
        goal=goal,
    )


def load_scenario(path: str) -> Scenario:
    folder = data_folder / "scenarios"
    with open(folder / f"{path}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    name = data.get("name", os.path.basename(path))

    agent_raw = data.get("agent", {})
    agent_cfg = AgentConfig(
        radius=float(agent_raw.get("radius", 0.25)),
        max_speed=float(agent_raw.get("max_speed", 1.6)),
        goal_attraction=float(agent_raw.get("goal_attraction", 3.0)),
        wall_repulsion=float(agent_raw.get("wall_repulsion", 6.0)),
        wall_range=float(agent_raw.get("wall_range", 1.5)),
        floor_repulsion=float(agent_raw.get("floor_repulsion", 6.0)),
        floor_range=float(agent_raw.get("floor_range", 0.7)),
        ramp_attraction=float(agent_raw.get("ramp_attraction", 2.5)),
        ramp_range=float(agent_raw.get("ramp_range", 4.0)),
        damping=float(agent_raw.get("damping", 0.7)),
    )

    sim_raw = data.get("simulation", {})
    sim_cfg = SimulationConfig(
        dt=float(sim_raw.get("dt", 0.05)),
        steps=int(sim_raw.get("steps", 1200)),
        seed=int(sim_raw.get("seed", 1337)),
    )

    groups = []
    for group in data.get("evacuees", []):
        count = int(group.get("count", 1))
        spawn = group.get("spawn", {})
        if "box" in spawn:
            smin = _vec(spawn["box"]["min"])
            smax = _vec(spawn["box"]["max"])
        else:
            smin = _vec(spawn.get("min", [0, 0, 0]))
            smax = _vec(spawn.get("max", [1, 1, 1]))

        rep = group.get("repulsion", {})
        anis = rep.get("anisotropy", {})
        rep_cfg = RepulsionConfig(
            strength=float(rep.get("strength", 2.5)),
            range=float(rep.get("range", 2.0)),
            anisotropy_enabled=bool(anis.get("enabled", False)),
            forward_strength=float(anis.get("forward_strength", 1.0)),
            backward_strength=float(anis.get("backward_strength", 1.0)),
            forward_angle_deg=float(anis.get("forward_angle_deg", 90.0)),
        )
        groups.append(
            ScenarioGroup(
                count=count, spawn_min=smin, spawn_max=smax, repulsion=rep_cfg
            )
        )

    return Scenario(name=name, groups=groups, agent_cfg=agent_cfg, sim_cfg=sim_cfg)


# ----------------------------- Utility Math -----------------------------


def clamp(x, a, b):
    return max(a, min(b, x))


def norm(v: Vector) -> float:
    return float(np.linalg.norm(v))


def unit(v: Vector) -> Vector:
    n = norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def sample_in_box(minv: Vector, maxv: Vector) -> Vector:
    return np.array(
        [
            random.uniform(minv[0], maxv[0]),
            random.uniform(minv[1], maxv[1]),
            random.uniform(minv[2], maxv[2]),
        ],
        dtype=float,
    )


def closest_point_on_aabb(point: Vector, minv: Vector, maxv: Vector) -> Vector:
    return np.array(
        [
            clamp(point[0], minv[0], maxv[0]),
            clamp(point[1], minv[1], maxv[1]),
            clamp(point[2], minv[2], maxv[2]),
        ],
        dtype=float,
    )


def point_in_aabb(point: Vector, minv: Vector, maxv: Vector) -> bool:
    return (
        minv[0] <= point[0] <= maxv[0]
        and minv[1] <= point[1] <= maxv[1]
        and minv[2] <= point[2] <= maxv[2]
    )


def closest_point_on_cylinder(
    point: Vector, center: Vector, radius: float, height: float
) -> Vector:
    # cylinder aligned with z-axis
    zmin = center[2]
    zmax = center[2] + height
    pz = clamp(point[2], zmin, zmax)
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    d = math.hypot(dx, dy)
    if d < 1e-8:
        cx, cy = center[0] + radius, center[1]
    else:
        cx = center[0] + radius * dx / d
        cy = center[1] + radius * dy / d
    return np.array([cx, cy, pz], dtype=float)


def closest_point_on_segment_xy(
    point: Vector, a: Vector, b: Vector
) -> Tuple[Vector, float, float]:
    ap = np.array([point[0] - a[0], point[1] - a[1]], dtype=float)
    ab = np.array([b[0] - a[0], b[1] - a[1]], dtype=float)
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-8:
        closest = np.array([a[0], a[1]], dtype=float)
        return closest, 0.0, float(np.linalg.norm(ap))
    t = clamp(float(np.dot(ap, ab)) / ab_len2, 0.0, 1.0)
    closest = np.array([a[0] + t * ab[0], a[1] + t * ab[1]], dtype=float)
    dist = float(np.linalg.norm(ap - t * ab))
    return closest, t, dist


# ----------------------------- Potentials -----------------------------


def attractive_force(pos: Vector, goal: Goal, strength: float) -> Vector:
    direction = goal.center - pos
    return strength * unit(direction)


def attraction_to_point(pos: Vector, p: Vector, strength: float, rng: float) -> Vector:
    dvec = p - pos
    d = norm(dvec)
    if d > rng or d < 1e-6:
        return np.zeros(3, dtype=float)
    mag = strength * (1 - d / rng)
    return mag * unit(dvec)


def attraction_to_ramp(
    pos: Vector, stair: Stair, strength: float, rng: float
) -> Vector:
    closest_xy, t, dist = closest_point_on_segment_xy(pos, stair.start, stair.end)
    if dist > rng:
        return np.zeros(3, dtype=float)

    target_z = float(stair.start[2] + t * (stair.end[2] - stair.start[2]))
    target_point = np.array([closest_xy[0], closest_xy[1], target_z], dtype=float)

    dvec = target_point - pos
    d = norm(dvec)
    if d < 1e-6:
        return np.zeros(3, dtype=float)

    lateral = np.array([dvec[0], dvec[1], 0.0], dtype=float)
    lateral_n = unit(lateral)
    slope_sign = 1.0 if (stair.end[2] - stair.start[2]) >= 0 else -1.0
    uphill = np.array(
        [stair.end[0] - stair.start[0], stair.end[1] - stair.start[1], 0.0], dtype=float
    )
    uphill = unit(uphill) * slope_sign

    mag = strength * (1 - d / rng)
    f = mag * unit(dvec)

    if norm(lateral_n) > 1e-6 and norm(uphill) > 1e-6:
        along = float(np.dot(lateral_n, uphill))
        f += (0.5 * mag * along) * np.array([uphill[0], uphill[1], 0.0], dtype=float)

    return f


def repulsion_from_point(pos: Vector, p: Vector, strength: float, rng: float) -> Vector:
    dvec = pos - p
    d = norm(dvec)
    if d > rng or d < 1e-6:
        return np.zeros(3, dtype=float)
    # smooth inverse-square falloff
    mag = strength * (1.0 / (d * d + 1e-6)) * (1 - d / rng)
    return mag * unit(dvec)


def repulsion_from_aabb(
    pos: Vector, minv: Vector, maxv: Vector, strength: float, rng: float
) -> Vector:
    cp = closest_point_on_aabb(pos, minv, maxv)
    return repulsion_from_point(pos, cp, strength, rng)


def repulsion_from_cylinder(
    pos: Vector,
    center: Vector,
    radius: float,
    height: float,
    strength: float,
    rng: float,
) -> Vector:
    cp = closest_point_on_cylinder(pos, center, radius, height)
    return repulsion_from_point(pos, cp, strength, rng)


def anisotropic_weight(
    forward_dir: Vector, direction_to_other: Vector, cfg: RepulsionConfig
) -> float:
    if not cfg.anisotropy_enabled:
        return 1.0
    fd = unit(forward_dir)
    od = unit(direction_to_other)
    if norm(fd) < 1e-6:
        return 1.0
    angle = math.degrees(math.acos(clamp(float(np.dot(fd, od)), -1.0, 1.0)))
    if angle <= cfg.forward_angle_deg:
        return cfg.forward_strength
    return cfg.backward_strength


# ----------------------------- Simulation -----------------------------


def assign_floor_z(pos: Vector, floors: List[Floor]) -> float:
    if not floors:
        return pos[2]
    # nearest floor by z
    z = pos[2]
    zs = [fl.z for fl in floors]
    idx = int(np.argmin([abs(z - zi) for zi in zs]))
    return zs[idx] + floors[idx].thickness * 0.5


def floor_thickness_at(z: float, floors: List[Floor]) -> float:
    for fl in floors:
        if abs(fl.z - z) < 1e-3:
            return fl.thickness * 0.5
    return 0.1


def ramp_projection(pos: Vector, stair: Stair):
    a = stair.start
    b = stair.end
    u = b - a
    length = norm(u)
    if length < 1e-6:
        return None
    u_dir = u / length
    v_dir = np.array([-u_dir[1], u_dir[0], 0.0], dtype=float)
    if norm(v_dir) < 1e-6:
        v_dir = np.array([0.0, 1.0, 0.0], dtype=float)
    v_dir = unit(v_dir)
    half_w = stair.width * 0.5

    rel = pos - a
    along = float(np.dot(rel, u_dir))
    lateral = float(np.dot(rel, v_dir))
    along_clamped = clamp(along, 0.0, length)
    lateral_clamped = clamp(lateral, -half_w, half_w)

    closest = a + along_clamped * u_dir + lateral_clamped * v_dir
    normal = np.cross(u_dir, v_dir)
    normal = unit(normal)
    if norm(normal) < 1e-6:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    normal_dist = float(
        np.dot(rel - (along_clamped * u_dir + lateral_clamped * v_dir), normal)
    )
    in_bounds = (0.0 <= along <= length) and (abs(lateral) <= half_w)
    return closest, normal, normal_dist, length, along, lateral, in_bounds


def ramp_surface_at(
    pos: Vector, floorplan: Floorplan
) -> Tuple[Optional[Vector], Optional[Vector]]:
    best_point = None
    best_normal = None
    best_abs = 1e18
    for st in floorplan.stairs:
        proj = ramp_projection(pos, st)
        if proj is None:
            continue
        closest, normal, normal_dist, length, along, lateral, in_bounds = proj
        abs_nd = abs(normal_dist)
        horiz_dist = math.hypot(closest[0] - pos[0], closest[1] - pos[1])

        if abs(lateral) > st.width * 0.35 and not in_bounds:
            continue
        if along < -st.width or along > length + st.width:
            continue
        if horiz_dist > st.width * 1.2 and not in_bounds:
            continue

        if abs_nd < best_abs:
            best_abs = abs_nd
            best_point = closest
            best_normal = normal

    return best_point, best_normal


def resolve_agent_z(pos: Vector, floorplan: Floorplan) -> float:
    ramp_point, _ = ramp_surface_at(pos, floorplan)
    if ramp_point is not None:
        return ramp_point[2]
    return assign_floor_z(pos, floorplan.floors)


def initialize_agents(scenario: Scenario, floors: List[Floor]) -> List[Agent]:
    agents = []
    for group in scenario.groups:
        for _ in range(group.count):
            p = sample_in_box(group.spawn_min, group.spawn_max)
            p[2] = assign_floor_z(p, floors)
            v = np.zeros(3, dtype=float)
            agents.append(
                Agent(
                    pos=p,
                    vel=v,
                    radius=scenario.agent_cfg.radius,
                    repulsion=group.repulsion,
                )
            )
    return agents


def compute_force(
    i: int,
    agents: List[Agent],
    floorplan: Floorplan,
    agent_cfg: AgentConfig,
) -> Vector:
    agent = agents[i]

    if not agent.active:
        return np.zeros(3, dtype=float)

    force = np.zeros(3, dtype=float)

    # blend goal force with ramp guidance so upper-floor agents prefer descent paths
    goal_vec = floorplan.goal.center - agent.pos
    goal_dist = norm(goal_vec)
    direct_goal_force = attractive_force(
        agent.pos, floorplan.goal, agent_cfg.goal_attraction
    )

    nearest_floor_z = assign_floor_z(agent.pos, floorplan.floors)
    on_goal_floor = abs(nearest_floor_z - floorplan.goal.center[2]) < 0.25
    if on_goal_floor:
        force += direct_goal_force
    else:
        # attenuate direct goal pull on upper floors to reduce wall-hugging local minima
        force += 0.2 * direct_goal_force

    # ramp attraction (project to ramp plane + downhill tangential push)
    if agent_cfg.ramp_attraction > 0 and floorplan.stairs:
        best_stair_force = np.zeros(3, dtype=float)
        best_score = -1e18
        for st in floorplan.stairs:
            proj = ramp_projection(agent.pos, st)
            if proj is None:
                continue
            closest, normal, normal_dist, length, along, lateral, in_bounds = proj
            half_w = st.width * 0.5
            lateral_pen = max(0.0, abs(lateral) - half_w)
            dist_to_plane = abs(normal_dist)
            to_closest = closest - agent.pos
            d = norm(to_closest)
            if d > agent_cfg.ramp_range * 1.6:
                continue

            score = (
                -dist_to_plane
                - 0.4 * lateral_pen
                - 0.1 * max(0.0, -along)
                - 0.1 * max(0.0, along - length)
            )
            if agent.pos[2] > floorplan.goal.center[2] + 0.25:
                score += 1.5
            if score > best_score:
                best_score = score
                mag = (
                    1.3
                    * agent_cfg.ramp_attraction
                    * (
                        1
                        - min(d, agent_cfg.ramp_range * 1.6)
                        / (agent_cfg.ramp_range * 1.6)
                    )
                )
                attraction = mag * unit(to_closest)
                slope = st.end[2] - st.start[2]
                downhill_dir = (st.end - st.start) / length
                if slope > 0:
                    downhill_dir = -downhill_dir
                downhill_push = (
                    0.9
                    * mag
                    * np.array([downhill_dir[0], downhill_dir[1], 0.0], dtype=float)
                )
                best_stair_force = attraction + downhill_push

        force += best_stair_force

    # walls and obstacles repulsion (skip if inside a mask)
    inside_mask = any(
        point_in_aabb(agent.pos, m.min, m.max) for m in floorplan.repulsion_masks
    )

    if not inside_mask:
        for wall in floorplan.walls:
            force += repulsion_from_aabb(
                agent.pos,
                wall.min,
                wall.max,
                agent_cfg.wall_repulsion,
                agent_cfg.wall_range,
            )

        for ob in floorplan.obstacles:
            if ob.kind == "box" and ob.min is not None and ob.max is not None:
                force += repulsion_from_aabb(
                    agent.pos,
                    ob.min,
                    ob.max,
                    agent_cfg.wall_repulsion,
                    agent_cfg.wall_range,
                )
            elif (
                ob.kind == "pillar"
                and ob.center is not None
                and ob.radius is not None
                and ob.height is not None
            ):
                force += repulsion_from_cylinder(
                    agent.pos,
                    ob.center,
                    ob.radius,
                    ob.height,
                    agent_cfg.wall_repulsion,
                    agent_cfg.wall_range,
                )

    # floor/ramp vertical stabilization for smooth z motion (avoid teleport-like hopping)
    ramp_point, ramp_normal = ramp_surface_at(agent.pos, floorplan)
    if ramp_point is not None and ramp_normal is not None:
        delta = ramp_point - agent.pos
        normal_component = float(np.dot(delta, ramp_normal))
        force += 10.0 * normal_component * ramp_normal
    elif floorplan.floors:
        target_z = assign_floor_z(agent.pos, floorplan.floors)
        dz = target_z - agent.pos[2]
        force += np.array([0.0, 0.0, 10.0 * dz], dtype=float)

    # agent-agent repulsion (disable emitter when that other agent has reached goal)
    for j, other in enumerate(agents):
        if i == j or not other.active or other.reached_goal:
            continue
        dvec = agent.pos - other.pos
        dist = norm(dvec)
        if dist < 1e-6:
            continue
        # anisotropic weighting based on other's velocity (who projects field)
        w = anisotropic_weight(other.vel, agent.pos - other.pos, other.repulsion)
        f = repulsion_from_point(
            agent.pos, other.pos, other.repulsion.strength * w, other.repulsion.range
        )
        # taper repulsion when either agent is near the goal to allow stacking at exit
        goal_near = (
            norm(agent.pos - floorplan.goal.center) < floorplan.goal.radius * 2.0
            or norm(other.pos - floorplan.goal.center) < floorplan.goal.radius * 2.0
        )
        if goal_near:
            f *= 0.1
        force += f

    return force


def step_simulation(
    agents: List[Agent],
    floorplan: Floorplan,
    agent_cfg: AgentConfig,
    dt: float,
):
    forces = [
        compute_force(i, agents, floorplan, agent_cfg) for i in range(len(agents))
    ]

    for i, agent in enumerate(agents):
        if not agent.active:
            continue
        desired_vel = forces[i]
        agent.vel = (
            1.0 - agent_cfg.damping
        ) * agent.vel + agent_cfg.damping * desired_vel
        # max speed
        speed = norm(agent.vel)
        if speed > agent_cfg.max_speed:
            agent.vel = agent.vel * (agent_cfg.max_speed / speed)

        agent.pos = agent.pos + agent.vel * dt

        # clamp to bounds
        bmin, bmax = floorplan.bounds
        agent.pos[0] = clamp(agent.pos[0], bmin[0], bmax[0])
        agent.pos[1] = clamp(agent.pos[1], bmin[1], bmax[1])

        # smooth z update instead of hard snapping each frame
        target_z = resolve_agent_z(agent.pos, floorplan)
        z_blend = clamp(0.35 + 6.0 * dt, 0.0, 1.0)
        agent.pos[2] = (1.0 - z_blend) * agent.pos[2] + z_blend * target_z

        # goal check
        if norm(agent.pos - floorplan.goal.center) <= floorplan.goal.radius:
            agent.active = False
            agent.reached_goal = True
            agent.vel[:] = 0.0
            agent.pos[:] = floorplan.goal.center.copy()


# ----------------------------- Visualization -----------------------------


def build_scene(floorplan: Floorplan):
    actors = []

    # bounds (wireframe)
    bmin, bmax = floorplan.bounds
    boundary = Box(
        pos=(bmin + bmax) * 0.5,
        length=bmax[0] - bmin[0],
        width=bmax[1] - bmin[1],
        height=bmax[2] - bmin[2],
    )
    boundary.wireframe().c("gray").alpha(0.05)
    actors.append(boundary)

    # floors
    for fl in floorplan.floors:
        z = fl.z + fl.thickness * 0.5
        floor_box = Box(
            pos=((bmin[0] + bmax[0]) * 0.5, (bmin[1] + bmax[1]) * 0.5, z),
            length=bmax[0] - bmin[0],
            width=bmax[1] - bmin[1],
            height=fl.thickness,
        )
        floor_box.c("lightgray").alpha(0.25)
        actors.append(floor_box)

    # walls
    for wall in floorplan.walls:
        center = (wall.min + wall.max) * 0.5
        dims = wall.max - wall.min
        wbox = Box(pos=center, length=dims[0], width=dims[1], height=dims[2])
        wbox.c("slategray").alpha(0.2)
        actors.append(wbox)

    # obstacles
    for ob in floorplan.obstacles:
        if ob.kind == "box" and ob.min is not None and ob.max is not None:
            center = (ob.min + ob.max) * 0.5
            dims = ob.max - ob.min
            obox = Box(pos=center, length=dims[0], width=dims[1], height=dims[2])
            obox.c("brown").alpha(0.35)
            actors.append(obox)
        elif (
            ob.kind == "pillar"
            and ob.center is not None
            and ob.radius is not None
            and ob.height is not None
        ):
            cyl = Cylinder(
                pos=(ob.center[0], ob.center[1], ob.center[2] + ob.height * 0.5),
                r=ob.radius,
                height=ob.height,
                axis=(0, 0, 1),
                c="tan",
                alpha=0.35,
            )
            actors.append(cyl)

    # stairs/ramps (more visible)
    for st in floorplan.stairs:
        dh = st.end - st.start
        run = float(np.linalg.norm(dh[:2]))
        if run < 1e-6:
            continue

        mid = (st.start + st.end) * 0.5
        yaw_deg = math.degrees(math.atan2(dh[1], dh[0]))
        slope_deg = math.degrees(math.atan2(dh[2], run))
        axis = np.array([-dh[1], dh[0], 0.0], dtype=float)
        axis_norm = float(np.linalg.norm(axis))

        ramp_body = Box(pos=mid, length=run, width=st.width, height=0.16)
        ramp_body.rotate(yaw_deg, axis=(0, 0, 1))
        if axis_norm > 1e-6:
            ramp_body.rotate(slope_deg, axis=axis / axis_norm)
        ramp_body.pos(mid)
        ramp_body.c("orange5").alpha(0.75)
        actors.append(ramp_body)

        ramp_outline = Box(pos=mid, length=run, width=st.width, height=0.18)
        ramp_outline.rotate(yaw_deg, axis=(0, 0, 1))
        if axis_norm > 1e-6:
            ramp_outline.rotate(slope_deg, axis=axis / axis_norm)
        ramp_outline.pos(mid)
        ramp_outline.wireframe().c("gold").alpha(0.9)
        actors.append(ramp_outline)

    # goal
    goal = Sphere(pos=floorplan.goal.center, r=floorplan.goal.radius)
    goal.c("green").alpha(0.6)
    actors.append(goal)

    return actors


def render_simulation(
    floorplan: Floorplan,
    scenario: Scenario,
    agents: List[Agent],
    steps: int,
    dt: float,
    output_video: Optional[str] = None,
):
    bmin, bmax = floorplan.bounds
    plotter = Plotter(title=floorplan.name, size=(1200, 800))
    plotter.show(*build_scene(floorplan), interactive=False)
    center = (bmin + bmax) * 0.5
    max_dim = float(np.max(bmax - bmin))
    cam_pos = center + np.array([1.2, 1.2, 1.0], dtype=float) * max_dim
    plotter.camera.SetPosition(cam_pos.tolist())
    plotter.camera.SetFocalPoint(center.tolist())
    plotter.camera.SetViewUp((0, 0, 1))
    plotter.camera.SetParallelProjection(True)

    # Vedo points for agents
    positions = np.array([a.pos for a in agents], dtype=float)
    pts = Points(positions, r=8)
    pts.c("blue")
    plotter.add(pts)

    status_text = Text2D("", pos="top-left", c="white", font="Courier", s=0.8)
    plotter.add(status_text)

    writer = None
    if output_video:
        writer = Video(output_video, fps=int(1.0 / dt))

    for step in range(steps):
        step_simulation(agents, floorplan, scenario.agent_cfg, dt)

        positions = np.array([a.pos for a in agents], dtype=float)
        pts.points = positions

        active_count = sum(1 for a in agents if a.active)
        reached_count = sum(1 for a in agents if a.reached_goal)
        status_text.text(
            f"step {step + 1}/{steps}   active: {active_count}   reached: {reached_count}"
        )

        plotter.render()

        if writer:
            writer.add_frame()

    if writer:
        writer.close()

    plotter.interactive().close()


# ----------------------------- Example Data -----------------------------


def make_example_data(root: str):
    os.makedirs(os.path.join(root, "data", "floorplans"), exist_ok=True)
    os.makedirs(os.path.join(root, "data", "scenarios"), exist_ok=True)

    floorplan = {
        "name": "Example 3-floor",
        "bounds": {"x": [0, 20], "y": [0, 12], "z": [0, 9]},
        "floors": [
            {"z": 0, "thickness": 0.2, "walkable": True},
            {"z": 3, "thickness": 0.2, "walkable": True},
            {"z": 6, "thickness": 0.2, "walkable": True},
        ],
        "walls": [
            {"min": [0, 0, 0], "max": [20, 0.0, 9]},
            {"min": [0, 12, 0], "max": [20, 12, 9]},
            {"min": [0, 0, 0], "max": [0, 12, 9]},
            {"min": [20, 0, 0], "max": [20, 12, 9]},
        ],
        "repulsion_masks": [
            # ramp openings
            {"min": [9, 3, 5.5], "max": [19, 5, 6.5]},
            {"min": [3, 7, 2.5], "max": [11, 9, 3.5]},
            # bottom exit opening near goal
            {"min": [-0.5, 5, 0], "max": [0.5, 7, 1]},
        ],
        "obstacles": [
            {"type": "box", "min": [6, 4, 0], "max": [8, 6, 3]},
            {"type": "pillar", "center": [14, 8, 0], "radius": 0.6, "height": 9},
        ],
        "stairs": [
            {"start": [18, 4, 6], "end": [10, 4, 3], "width": 3.5},
            {"start": [4, 8, 3], "end": [10, 8, 0], "width": 3.5},
        ],
        "goal": {"center": [0, 6, 0], "radius": 0.7},
    }

    scenario = {
        "name": "Mixed crowd",
        "evacuees": [
            {
                "count": 35,
                "spawn": {"box": {"min": [2, 2, 0.2], "max": [18, 10, 2.8]}},
                "repulsion": {
                    "strength": 2.5,
                    "range": 2.2,
                    "anisotropy": {
                        "enabled": True,
                        "forward_strength": 1.5,
                        "backward_strength": 0.5,
                        "forward_angle_deg": 70,
                    },
                },
            },
            {
                "count": 15,
                "spawn": {"box": {"min": [2, 2, 3.2], "max": [18, 10, 5.8]}},
                "repulsion": {
                    "strength": 2.0,
                    "range": 2.0,
                    "anisotropy": {"enabled": False},
                },
            },
        ],
        "agent": {
            "radius": 0.25,
            "max_speed": 1.6,
            "goal_attraction": 3.5,
            "wall_repulsion": 6.0,
            "wall_range": 1.5,
            "floor_repulsion": 6.0,
            "floor_range": 0.7,
            "ramp_attraction": 2.5,
            "ramp_range": 4.0,
            "damping": 0.7,
        },
        "simulation": {"dt": 0.05, "steps": 1200, "seed": 1337},
    }

    fp_path = os.path.join(root, "data", "floorplans", "example.json")
    sc_path = os.path.join(root, "data", "scenarios", "example.json")

    with open(fp_path, "w", encoding="utf-8") as f:
        json.dump(floorplan, f, indent=2)

    with open(sc_path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2)

    print(f"Wrote example floorplan: {fp_path}")
    print(f"Wrote example scenario: {sc_path}")


# ----------------------------- Main -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradient-descent evacuation simulation with vedo visualization."
    )
    parser.add_argument(
        "--floorplan", type=str, default=None, help="Path to floorplan JSON"
    )
    parser.add_argument(
        "--scenario", type=str, default=None, help="Path to scenario JSON"
    )
    parser.add_argument("--steps", type=int, default=None, help="Override steps")
    parser.add_argument("--dt", type=float, default=None, help="Override dt")
    parser.add_argument(
        "--video", type=str, default=None, help="Output video file (e.g., out.mp4)"
    )
    parser.add_argument(
        "--make-example-data",
        action="store_true",
        help="Create example data in data/ folder",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root = os.path.dirname(os.path.abspath(__file__))

    if args.make_example_data:
        make_example_data(root)
        return

    if not args.floorplan or not args.scenario:
        print("Please provide --floorplan and --scenario (or use --make-example-data).")
        return

    floorplan = load_floorplan(args.floorplan)
    scenario = load_scenario(args.scenario)

    if args.steps is not None:
        scenario.sim_cfg.steps = int(args.steps)
    if args.dt is not None:
        scenario.sim_cfg.dt = float(args.dt)

    random.seed(scenario.sim_cfg.seed)
    np.random.seed(scenario.sim_cfg.seed)

    agents = initialize_agents(scenario, floorplan.floors)

    render_simulation(
        floorplan=floorplan,
        scenario=scenario,
        agents=agents,
        steps=scenario.sim_cfg.steps,
        dt=scenario.sim_cfg.dt,
        output_video=args.video,
    )


if __name__ == "__main__":
    main()
