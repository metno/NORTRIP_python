from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_plot_style() -> None:
    """Apply a clean, readable default style for all NORTRIP plots.

    - Prefer Matplotlib's bundled seaborn-like style for sensible defaults
    - Use a colorblind-friendly palette
    - Make grids subtle, remove top/right spines
    - Harmonize font sizes and line widths
    """

    # Use a pleasant base style if available
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        plt.style.use("default")

    # Core rcParams tuned for clarity
    rc_updates = {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "figure.figsize": [8, 6],
        "font.size": 6,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.color": "#cccccc",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "font.family": "DejaVu Sans",
    }
    mpl.rcParams.update(rc_updates)

    # Color cycle (colorblind-friendly, Matplotlib default-like but explicit)
    try:
        from cycler import cycler

        mpl.rcParams["axes.prop_cycle"] = cycler(
            color=[
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
        )
    except Exception:
        # cycler is a matplotlib dependency; just ignore if unavailable
        pass


def shrink_y_labels(axes, points: int = 2) -> None:
    """Reduce y-axis label font size by a given number of points.

    Accepts a single Axes or an iterable of Axes.
    """
    try:
        from collections.abc import Iterable

        axes_iter = axes if isinstance(axes, Iterable) else [axes]
    except Exception:
        axes_iter = [axes]

    for ax in axes_iter:
        if ax is None:
            continue
        label = ax.yaxis.get_label()
        try:
            current = float(label.get_size())
        except Exception:
            current = mpl.rcParams.get("axes.labelsize", 8)
        new_size = max(1.0, current - float(points))
        label.set_size(new_size)
