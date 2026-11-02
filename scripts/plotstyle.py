"""Common look of the benchmark graphs (matplotlib, PNG).

Series colours always come in the same fixed order (colour-blind safe for
neighbouring series); a series keeps its colour from one graph to the next.
Train curves are dashed and validation curves solid, so the two never rely on
colour alone.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9890"


def color(i):
    if i >= len(SERIES):
        raise ValueError(f"at most {len(SERIES)} series per graph: split it")
    return SERIES[i]


def figure(n_panels=1, width=6.0, height=4.2):
    plt.rcParams.update({
        "font.size": 10,
        "axes.edgecolor": NEUTRAL,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    })
    fig, axes = plt.subplots(1, n_panels, figsize=(width * n_panels, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[0]:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig, list(axes[0])


def log_count_axis(ax):
    """Log-scale x axis labelled 50k, 100k, 200k... instead of powers of ten."""
    ax.set_xscale("log")
    label = FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k")
    ax.xaxis.set_major_formatter(label)
    ax.xaxis.set_minor_formatter(label)


def save(fig, path, title=None):
    if title:
        fig.suptitle(title, color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
