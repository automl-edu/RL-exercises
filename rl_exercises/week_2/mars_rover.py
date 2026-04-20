from __future__ import annotations

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # type: ignore[import]
from rich import print as printr
from rl_exercises.environments import MarsRover

env = MarsRover()
actions = [0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]

states = []
s, info = env.reset()
states.append(s)
for i in range(env.horizon):
    action = actions[i]
    s_next, r, terminated, truncated, info = env.step(action)
    printr(
        f"Step: {i}, state: {s}, action: {action}, next state: {s_next}, "
        f"reward: {r}, terminated: {terminated}, truncated: {truncated}"
    )
    s = s_next
    states.append(s)

# Plot
fig, ax = plt.subplots()
image = plt.imread("figures/alien_1f47d.png")
image_box = OffsetImage(image, zoom=0.1)
x = np.arange(0, len(states))
y = states
for x0, y0 in zip(x, y):
    ab = AnnotationBbox(image_box, (x0, y0), frameon=False)
    ax.add_artist(ab)
ax.plot(x, y, c="green")
ax.set_xlabel("Step")
ax.set_ylabel("State")
plt.show()
