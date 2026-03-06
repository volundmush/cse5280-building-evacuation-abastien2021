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
    {"min": [9,5,0], "max": [11,7,3], "target_z": 0},
    {"min": [9,5,3], "max": [11,7,6], "target_z": 3}
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
class Stair:
    min: Vector
    max: Vector
    target_z: float


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


# ----------------------------- JSON Loading -----------------------------


def _vec(v) -> Vector:
    return np.array(v, dtype=float)


def load_floorplan(path: str) -> Floorplan:
    with open(path, "r", encoding="utf-8") as f:
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

    stairs = []
    for st in data.get("stairs", []):
        stairs.append(
            Stair(
                min=_vec(st["min"]),
                max=_vec(st["max"]),
                target_z=float(st["target_z"]),
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
        stairs=stairs,
        goal=goal,
    )


def load_scenario(path: str) -> Scenario:
    with open(path, "r", encoding="utf-8") as f:
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


# ----------------------------- Potentials -----------------------------


def attractive_force(pos: Vector, goal: Goal, strength: float) -> Vector:
    direction = goal.center - pos
    return strength * unit(direction)


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


def stair_target_z(pos: Vector, floorplan: Floorplan) -> Optional[float]:
    candidates = []
    current_floor = pos[2]
    if floorplan.floors:
        floor_levels = [fl.z for fl in floorplan.floors]
        current_floor = min(floor_levels, key=lambda z: abs(z - pos[2]))
    for st in floorplan.stairs:
        if st.min[0] <= pos[0] <= st.max[0] and st.min[1] <= pos[1] <= st.max[1]:
            target = st.target_z
            if floorplan.floors:
                floor_levels = [fl.z for fl in floorplan.floors]
                target = min(floor_levels, key=lambda z: abs(z - target))
            if target < current_floor - 1e-3:
                candidates.append(target)
    if not candidates:
        return None
    return max(candidates)


def resolve_agent_z(pos: Vector, floorplan: Floorplan) -> float:
    target = stair_target_z(pos, floorplan)
    if target is not None:
        return target + floor_thickness_at(target, floorplan.floors)
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

    # goal attraction
    force += attractive_force(agent.pos, floorplan.goal, agent_cfg.goal_attraction)

    # walls and obstacles repulsion
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

    # floor repulsion: keep on current floor plane (soft)
    if floorplan.floors and stair_target_z(agent.pos, floorplan) is None:
        target_z = assign_floor_z(agent.pos, floorplan.floors)
        dz = agent.pos[2] - target_z
        if abs(dz) > 1e-4:
            # pull back toward plane
            floor_point = np.array([agent.pos[0], agent.pos[1], target_z], dtype=float)
            force += repulsion_from_point(
                agent.pos, floor_point, agent_cfg.floor_repulsion, agent_cfg.floor_range
            ) * (-1.0)

    # agent-agent repulsion (if other active)
    for j, other in enumerate(agents):
        if i == j or not other.active:
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
        agent.pos[2] = resolve_agent_z(agent.pos, floorplan)

        # goal check
        if norm(agent.pos - floorplan.goal.center) <= floorplan.goal.radius:
            agent.active = False
            agent.vel[:] = 0.0


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

    # stairs
    for st in floorplan.stairs:
        center = (st.min + st.max) * 0.5
        dims = st.max - st.min
        sbox = Box(pos=center, length=dims[0], width=dims[1], height=dims[2])
        sbox.c("orange").alpha(0.35)
        actors.append(sbox)

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
        status_text.text(f"step {step + 1}/{steps}   active: {active_count}")

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
            {"min": [0, 0, 0], "max": [20, 0.3, 9]},
            {"min": [0, 11.7, 0], "max": [20, 12, 9]},
            {"min": [0, 0, 0], "max": [0.3, 12, 9]},
            {"min": [19.7, 0, 0], "max": [20, 12, 9]},
        ],
        "obstacles": [
            {"type": "box", "min": [6, 4, 0], "max": [8, 6, 3]},
            {"type": "pillar", "center": [14, 8, 0], "radius": 0.6, "height": 9},
        ],
        "stairs": [
            {"min": [9, 5, 0], "max": [11, 7, 3], "target_z": 0},
            {"min": [9, 5, 3], "max": [11, 7, 6], "target_z": 3},
        ],
        "goal": {"center": [10, 0.8, 0], "radius": 0.7},
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
