# CSE5280: Building Evacuation Simulation by Gradient Descent
## Author: Andrew Bastien

# Overview
Building on previous assignments, now we have a multi-floor simulation space with obstacles and vertical movement where the movement now cares about obstacles and ramps in addition to walls and other evacuees.

## Setup
`python -m venv .venv`
Linux: `source .venv/bin/activate` - or Windows: `.\Scripts\activate.bat`
`pip install -r requirements.txt`

## Why is this not a Jupyter notebook?
Because the default rendering of the simulations is interactive and I couldn't figure out how to preseve that in a Jupyter Notebook. Might be issues with running them on my Bazzite, not sure. Wanted to prioritize replicable reliable functionality.

## Project Files:
- the `data/` directory contains simulations to run.
- `simulation.py` is the engine
- various `.mp4` files because they turned out small and didn't need to be externally hosted.

# Summary of Challenges and Solutions, Observations and Insights
Each agent operates independently from its current position and reacts only to local attractive and repulsive influences.

In testing and exploration of the problem, several major challenges were discovered while attempting to apply Gradient Descend "pathfinding" to a multi-floored navigation. The most notable headaches were:

**You can't vibrate through the floor, Agent 42!**: Agents on higher floors would get stuck if reaching the ramp took them further from the goal. After a great deal of thinking, I concluded that a pure gradient descent approach maps poorly to 3D. There has to be a mechanism to account for situations like "the stairs down are on the opposite end of the building from the exit, ignoring Z coordinates." Gradient descent approach modified so that ramps leading towards the goal are a local goal. This feels like a betrayal of the gradient descent concept, but I don't believe that it's possible without such a mechanism.

**It's too tight I can't squeeze through!**: Obstacles too close to one another, especially when combined with Agent social pressures, would create local minima and deadlock zones. There really isn't any while remaining pure gradient descent. Some wall following works for some of it.

**How do I punch a hole in the walls and floors?**: Ramps are kind of a poor man's way of modeling stairs, but even then they're not intuitive on a technical level. For instance, there has to be an opening in the floor for an Agent to enter the ramp from or else the floor's collision will keep them from getting on the ramp. I considered using complex polygonal shapes instead of simple boxes for floors and walls, but then had the brilliant idea to simply add polygons I call Openings to the mix. They aren't visually rendered, but where they overlap with a fixture (wall, floor, obstacle, etc), they disable its repulsive force, which effectively disables the collision. Getting this right proved rather tricky; early builds had Agents wandering towards a Ramp and then sort of plunging straight down once they were "above the ramp", effectively skipping any obvious "go down the stairs" phase.

**The Agents are acting like Lemmings and jumping off Cliffs to their dooms...**: It turns out that working with ramps is very tricky. Agents that reached the ramp would then path straight towards the next ramp and... off the side of their current ramp and immediately drop to the floor below. These Agents are not expected to be ninjas, so the sides of ramps had to become impassable.

**No, Agent 42, please don't get back on the ramp!**: Sometimes when an agent finished navigating a ramp and reached a new floor, they would then turn right back around and try to climb it again while moving towards their next local goal. This would create head-of-line blocking and a deadlock with other agents still using that ramp. So logic had to be added so a ramp is only entered if using it brings one closer to the goal floor.

**Stop bunching up at the exit, the door's not locked!**: This is a simulation of evacuating of a building. Is the building on fire? I forget why it's being evacuated. Either way, a mechanism had to be introduced to disable the collision/repulsion on an Agent that had reached the goal. This allows them to "exit the simulation" once they reach it and stop blocking others from progressing.

# Algorithm Overview

At a high level, each agent follows a locally computed force model. There is no global shortest-path solution or precomputed route tree. Instead, motion at time step $t$ is driven by a sum of attractive and repulsive terms evaluated from the agent's current state.

Let the state of agent $i$ be

$$
\mathbf{x}_i(t) = [x_i(t), y_i(t), z_i(t)]^\top, \qquad \mathbf{v}_i(t) = \dot{\mathbf{x}}_i(t).
$$

The total steering force is modeled as

$$
\mathbf{F}_i = \mathbf{F}_{\text{nav},i} + \mathbf{F}_{\text{wall},i} + \mathbf{F}_{\text{social},i} - \gamma \mathbf{v}_i,
$$

where:
- $\mathbf{F}_{\text{nav},i}$ attracts the agent toward either the exit or the locally relevant ramp entrance,
- $\mathbf{F}_{\text{wall},i}$ repels the agent from nearby blocked geometry,
- $\mathbf{F}_{\text{social},i}$ repels the agent from other evacuees,
- and $\gamma \mathbf{v}_i$ is a damping term that suppresses oscillation.

The desired local navigation direction is

$$
\hat{\mathbf{d}}_i = \frac{\mathbf{g}_i - \mathbf{x}_{i,xy}}{\left\|\mathbf{g}_i - \mathbf{x}_{i,xy}\right\| + \varepsilon},
$$

so the attraction term becomes

$$
\mathbf{F}_{\text{nav},i} = k_{\text{nav}} \hat{\mathbf{d}}_i.
$$

Here $\mathbf{g}_i$ is not a globally planned waypoint list. It is chosen only from immediate context:

$$
\mathbf{g}_i =
\begin{cases}
\mathbf{g}_{\text{exit}}, & \text{if the agent is already on the goal floor},\\[4pt]
\mathbf{g}_{\text{ramp-entry}}, & \text{if a ramp on the current floor moves the agent closer to the goal floor.}
\end{cases}
$$

Wall avoidance is based on a distance field over blocked cells. If $d_i$ is the sampled distance from the agent to the nearest blocked region, then a simple repulsive form is

$$
\mathbf{F}_{\text{wall},i} = k_{\text{wall}} \, \max\!\left(0, 1 - \frac{d_i}{r_{\text{wall}}}\right) \hat{\mathbf{n}}_i,
$$

where $\hat{\mathbf{n}}_i$ points away from the nearest obstacle boundary.

Social repulsion is accumulated over nearby agents:

$$
\mathbf{F}_{\text{social},i} = \sum_{j \neq i}
k_{\text{social}} \, \phi\!\left(\lVert \mathbf{x}_{i,xy} - \mathbf{x}_{j,xy} \rVert\right)
\hat{\mathbf{u}}_{ji},
$$

with $\hat{\mathbf{u}}_{ji}$ pointing from agent $j$ to agent $i$ and $\phi(\cdot)$ decaying with distance. In anisotropic cases, the contribution is weighted more strongly for neighbors in front of the agent than behind it.

Once an agent has entered a ramp, the free-motion rule is replaced by a constrained ramp-following rule. The agent is projected into the ramp corridor, steered along the ramp tangent, and repelled from the ramp edges so it cannot "cut across" the slope or fall off the sides. Ramp entry is one-way with respect to the goal floor: the valid entry side is whichever endpoint decreases $|f - f_{\text{goal}}|$.

The simulation then advances with explicit Euler integration:

$$
\mathbf{v}_i(t + \Delta t) = \operatorname{clip}\!\left(\mathbf{v}_i(t) + \Delta t \, \mathbf{F}_i,\; v_{\max}\right),
$$

$$
\mathbf{x}_i(t + \Delta t) = \mathbf{x}_i(t) + \Delta t \, \mathbf{v}_i(t + \Delta t).
$$

In pseudo-code, one update step is:

```text
for each active agent i:
	update whether i is on a floor or on a ramp

	if i has reached the goal region:
		deactivate i
		continue

	choose local target g_i:
		if floor(i) == goal_floor:
			g_i <- exit center
		else:
			g_i <- nearest ramp entrance that moves closer to goal_floor

	compute navigation force toward g_i
	compute wall repulsion from local distance-to-block samples
	compute social repulsion from neighboring agents

	if i is on a ramp:
		replace free steering with ramp-tangent steering
		clamp lateral position to ramp width
		apply edge repulsion so i stays on the ramp

	F_i <- F_nav + F_wall + F_social - damping * v_i
	v_i <- speed-limited(v_i + dt * F_i)
	x_i <- x_i + dt * v_i
```

This keeps the implementation close to the spirit of gradient descent: agents respond to local forces and local geometry, while ramps act as special traversable regions that redirect motion between floors.


# Scenarios:
The four scenarios are:
`python simulation.py --floorplan basic --scenario basic`
`python simulation.py --floorplan atrium_loop --scenario atrium_loop`
`python simulation.py --floorplan offset_corners --scenario offset_corners`
`python simulation.py --floorplan four_floor_switchback --scenario four_floor_switchback`
`python simulation.py --floorplan sublevels --scenario sublevels`

A fourth floor variant was included to demonstrate that algorithmic pathfinding is modular. A sublevels variant shows that it works for ascending too.