"""Shared matplotlib style (reference categorical palette, thin marks, recessive grid)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# fixed categorical order (identity follows entity, never rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEIZURE_SHADE = "#e34948"

# stable colors per configuration so every figure uses the same identity
CONFIG_COLORS = {
    "RF-all18": SERIES[0],
    "CNN-all18": SERIES[1],
    "RF-top8": SERIES[2],
    "RF-top4": SERIES[6],
}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": TEXT2,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


apply_style()
