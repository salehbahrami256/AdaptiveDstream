from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_state(model, recent_points=None, title=None, save_path=None, axes=(0, 1)):
    """Visualize the current grid/cluster state.

    Drawing every axis-aligned leaf as a rectangle only makes unambiguous
    sense up to 2D:

    - ``dim == 1``: leaves are drawn as intervals along a single axis.
    - ``dim == 2``: the exact grid partition, as rectangles (original
      behavior).
    - ``dim >= 3``: a hyper-rectangle projected onto two axes overlaps with
      every other leaf that shares those two axes' bounds, so drawing "the
      grid" in 2D is misleading rather than merely imprecise. Instead this
      renders a scatter of points colored by their *predicted cluster*,
      projected onto ``axes`` (default: the first two dimensions) — which
      stays meaningful in any dimension — annotated with the leaf/dense/
      cluster counts that the rectangle view would otherwise have conveyed.
    """
    model._assign_clusters()
    if model.dim == 1:
        return _plot_state_1d(model, recent_points, title, save_path)
    if model.dim == 2:
        return _plot_state_2d(model, recent_points, title, save_path)
    return _plot_state_projection(model, recent_points, title, save_path, axes)


def _plot_state_2d(model, recent_points, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    if recent_points is not None and len(recent_points):
        ax.scatter(recent_points[:, 0], recent_points[:, 1], s=8, alpha=0.35)
    for cell in model.leaves():
        state = cell.state(model.t, model.decay, model.dense_threshold, model.sparse_threshold)
        rect = Rectangle(
            (cell.lower[0], cell.lower[1]),
            cell.upper[0] - cell.lower[0],
            cell.upper[1] - cell.lower[1],
            fill=False,
            linewidth=1.7 if state == "dense" else 0.7,
            alpha=0.8 if state == "dense" else 0.35,
        )
        ax.add_patch(rect)
        if cell.cluster_id is not None:
            cx, cy = cell.center
            ax.text(cx, cy, str(cell.cluster_id), ha="center", va="center", fontsize=8)
    ax.set_xlim(model.lower[0], model.upper[0])
    ax.set_ylim(model.lower[1], model.upper[1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title or f"Adaptive D-Stream at t={model.t}")
    ax.set_aspect("equal", adjustable="box")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def _plot_state_1d(model, recent_points, title, save_path):
    fig, ax = plt.subplots(figsize=(9, 3))
    y_cells, y_points = 0.6, 0.2
    for cell in model.leaves():
        state = cell.state(model.t, model.decay, model.dense_threshold, model.sparse_threshold)
        ax.plot(
            [cell.lower[0], cell.upper[0]], [y_cells, y_cells],
            color="tab:blue",
            linewidth=6 if state == "dense" else 2.5,
            alpha=0.85 if state == "dense" else 0.35,
            solid_capstyle="butt",
        )
        if cell.cluster_id is not None:
            ax.text(cell.center[0], y_cells + 0.08, str(cell.cluster_id), ha="center", fontsize=8)
    if recent_points is not None and len(recent_points):
        ax.scatter(recent_points[:, 0], np.full(len(recent_points), y_points), s=8, alpha=0.35)
    ax.set_xlim(model.lower[0], model.upper[0])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("x1")
    ax.set_title(title or f"Adaptive D-Stream at t={model.t}")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def _plot_state_projection(model, recent_points, title, save_path, axes):
    i, j = axes
    fig, ax = plt.subplots(figsize=(8, 8))
    if recent_points is not None and len(recent_points):
        preds = [model.predict_point_cluster(x) for x in recent_points]
        unassigned = np.array([p is None for p in preds])
        if unassigned.any():
            pts = recent_points[unassigned]
            ax.scatter(pts[:, i], pts[:, j], s=8, alpha=0.25, color="gray", label="unassigned")
        assigned = ~unassigned
        if assigned.any():
            pts = recent_points[assigned]
            labels = np.array([p for p in preds if p is not None], dtype=int)
            scatter = ax.scatter(pts[:, i], pts[:, j], s=12, alpha=0.7, c=labels, cmap="tab10")
            handles, _ = scatter.legend_elements()
            if handles:
                ax.legend(handles, sorted(set(labels.tolist())), title="cluster", fontsize=7, loc="best")
    ax.set_xlim(model.lower[i], model.upper[i])
    ax.set_ylim(model.lower[j], model.upper[j])
    ax.set_xlabel(f"x{i + 1}")
    ax.set_ylabel(f"x{j + 1}")
    s = model.summary()
    subtitle = (f"dim={model.dim}, projected onto axes ({i + 1}, {j + 1})\n"
                f"leaves={s['n_leaves']}  dense={s['n_dense_leaves']}  clusters={s['n_clusters']}")
    ax.set_title(f"{title or f'Adaptive D-Stream at t={model.t}'}\n{subtitle}", fontsize=10)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
