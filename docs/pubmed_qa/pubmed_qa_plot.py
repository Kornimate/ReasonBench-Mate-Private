from matplotlib import pyplot as plt
import numpy as np
import os

if not os.path.exists("results"):
    os.makedirs("results")

methods = ["cot_sc", "cot", "foa", "got", "heterogeneous_foa", "io", "reagents", "rap", "react", "tot_bfs", "tot_dfs"]

bar_values = [0.6, 0.6, 0.9, 0.4, 0.8, 0.5, 0.8, 0.8, 0.8, 0.9, 0.3]

hetmap_values = np.array([
    [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]).T

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(hetmap_values, aspect="equal", cmap="viridis", vmin=0, vmax=1)

cbar = fig.colorbar(ax.images[0], label="Value", ticks=[0, 1], orientation="vertical")
cbar.set_ticklabels(["Not solved", "Solved"])

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha="right")
ax.set_yticks(range(hetmap_values.shape[0]))
ax.set_yticklabels([f"Run {i+1}" for i in range(hetmap_values.shape[0])])

ax.set_ylabel("Run / sample")
ax.set_xlabel("Method")
ax.set_title("Results for 10 methods")

ax.set_xticks(np.arange(-0.5, hetmap_values.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, hetmap_values.shape[0], 1), minor=True)
ax.grid(which="minor", axis="both", color="white", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()
plt.savefig("results/methods_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

plt.bar(methods, bar_values)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Method")
plt.ylabel("Value")
plt.title("Bar Chart of Method AVG Values of 10 runs")
plt.savefig("results/methods_bar.png")
plt.show()
