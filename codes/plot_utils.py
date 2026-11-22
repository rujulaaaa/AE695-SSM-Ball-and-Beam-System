"""
plot_utils.py -- shared plot style plus a few helpers for saving stuff.

Figures go to ./figures as 200-dpi PNGs (looks sharp enough in the report
PDF), and the text tables/results go to ./results.
"""
import os

import matplotlib
matplotlib.use("Agg")                      # we only save files, so no window needed
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.6,
})


def savefig(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_result(name: str, text: str) -> str:
    path = os.path.join(RES_DIR, name)
    with open(path, "w") as fh:
        fh.write(text.rstrip() + "\n")
    return path


def fmt(M, prec: int = 4) -> str:
    return np.array2string(np.asarray(M), precision=prec, suppress_small=True,
                           max_line_width=130)
