"""Shared plotting helpers for clustering outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt


class ClusterPlotter:
    @staticmethod
    def draw_cluster_panel(ax, series_dataset, labels, center, cluster_id: int, x_limit: int, y_limit=None, title=None):
        for xx in series_dataset[labels == cluster_id]:
            ax.plot(xx.ravel(), 'k-', alpha=0.2)
        if center is not None:
            ax.plot(center.ravel(), 'r-')
        ax.set_xlim(0, x_limit)
        if y_limit is not None:
            ax.set_ylim(*y_limit)
        ax.text(0.55, 0.85, f'Cluster {cluster_id + 1}', transform=ax.transAxes)
        if title:
            ax.set_title(title)

    @staticmethod
    def new_figure(rows: int, cols: int):
        fig = plt.figure()
        return fig, rows, cols
