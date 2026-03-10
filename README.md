# CSE5280: Building Evacuation Simulation by Gradient Descent
## Author: Andrew Bastien

# Overview
Building on previous assignments, now we have a multi-floor simulation space with obstacles and vertical movement where the movement now cares about obstacles and ramps in addition to walls and other evacuees.

# Project Files:
- the `data/` directory contains simulations to run.
- `simulation.py` is the engine

# Summary of Challenges and Solutions, Observations and Insights
Each agent operates independently from its current position and reacts only to local attractive and repulsive influences.

In testing and exploration of the problem, several major challenges were discovered while attempting to apply Gradient Descend "pathfinding" to a multi-floored navigation. The most notable headaches were:

**You can't vibrate through the floor, Agent 42!**: Agents on higher floors would get stuck if reaching the ramp took them further from the goal. After a great deal of thinking, I concluded that a pure gradient descent approach maps poorly to 3D. There has to be a mechanism to account for situations like "the stairs down are on the opposite end of the building from the exit, ignoring Z coordinates." Gradient descent approach modified so that ramps leading towards the goal are a local goal. This feels like a betrayal of the gradient descent concept, but I don't believe that it's possible without such a mechanism.

**It's too tight I can't squeeze through!**: Obstacles too close to one another, especially when combined with Agent social pressures, would create local minima and deadlock zones. There really isn't any while remaining pure gradient descent. Some wall following works for some of it.

**How do I punch a hole in the walls and floors?**: Ramps are kind of a poor man's way of modeling stairs, but even then they're not intuitive on a technical level. For instance, there has to be an opening in the floor for an Agent to enter the ramp from or else the floor's collision will keep them from getting on the ramp. I considered using complex polygonal shapes instead of simple boxes for floors and walls, but then had the brilliant idea to simply add polygons I call Openings to the mix. They aren't visually rendered, but where they overlap with a fixture (wall, floor, obstacle, etc), they disable its repulsive force, which effectively disables the collision. Getting this right proved rather tricky; early builds had Agents wandering towards a Ramp and then sort of plunging straight down once they were "above the ramp", effectively skipping any obvious "go down the stairs" phase.

**The Agents are acting like Lemmings and jumping off Cliffs to their dooms...**: It turns out that working with ramps is very tricky. Agents that reached the ramp would then path straight towards the next ramp and... off the side of their current ramp and immediately drop to the floor below. These Agents are not expected to be ninjas, so the sides of ramps had to become impassable.

**No, Agent 42, please don't get back on the ramp!**: Sometimes when an agent finished navigating a ramp and reached a new floor, they would then turn right back around and try to ascend it while moving towards their next local goal. This would create head-of-line-blocking and a deadlock with other agents still descending. So logic had to be added so ramps are one-way and can only be entered if they would take one closer to the end goal.

**Stop bunching up at the exit, the door's not locked!**: This is a simulation of evacuating of a building. Is the building on fire? I forget why it's being evacuated. Either way, a mechanism had to be introduced to disable the collision/repulsion on an Agent that had reached the goal. This allows them to "exit the simulation" once they reach it and stop blocking others from progressing.

# Scenarios:
The four scenarios are:
`python simulation.py --floorplan basic --scenario basic`
`python simulation.py --floorplan atrium_loop --scenario atrium_loop`
`python simulation.py --floorplan offset_corners --scenario offset_corners`
`python simulation.py --floorplan four_floor_switchback --scenario four_floor_switchback`

A fourth floor variant was included to demonstrate that algorithmic pathfinding is modular.