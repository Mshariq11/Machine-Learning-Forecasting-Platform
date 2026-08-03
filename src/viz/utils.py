"""
Visualization Utilities
"""

import matplotlib.pyplot as plt

from .style import *


def apply_style(ax, title="", xlabel="", ylabel=""):
    """
    Apply consistent styling.
    """

    ax.set_title(
        title,
        fontsize=TITLE_SIZE,
        fontweight="bold"
    )

    ax.set_xlabel(
        xlabel,
        fontsize=LABEL_SIZE
    )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_SIZE
    )

    ax.grid(
        alpha=GRID_ALPHA
    )

    plt.tight_layout()