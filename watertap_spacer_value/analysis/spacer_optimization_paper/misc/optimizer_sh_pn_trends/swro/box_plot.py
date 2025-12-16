import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------
#  FIXED COLORS & MARKERS FOR CORRELATIONS (ORDERED)
# ---------------------------------------------------------
colors = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677"]
marker_styles = ["D", "o", "^", "v", "s"]  # diamond, circle, up-triangle, down-triangle, square

correlation_map = {
    "daCosta": {"color": colors[0], "legend": "Da Costa et al."},
    "Guillen": {"color": colors[1], "legend": "Guillen et al."},
    "Koustou": {"color": colors[2], "legend": "Koustou et al."},
    "Kuroda": {"color": colors[3], "legend": "Kuroda et al."},
    "Schock": {"color": colors[4], "legend": "Schock et al."},
}

ordered_corrs = list(correlation_map.keys())  # enforce plotting order

sheets = ["SWRO", "BWRO"]

# ---------------------------------------------------------
#  LOOP THROUGH EXCEL SHEETS
# ---------------------------------------------------------
for sheet in sheets:
    df = pd.read_excel("Overall_data.xlsx", sheet_name=sheet)

    types = df["Type"].astype(str).tolist()
    corr_cols = ordered_corrs                     # override Excel order
    data = df[corr_cols].values                  # numeric data in fixed order

    # ---------------------------------------------------------
    #  Matplotlib style
    # ---------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": "Arial",
        "mathtext.default": "regular",
        "svg.fonttype": "none",
    })
    plt.style.use("seaborn-white")
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---------------------------------------------------------
    #  1. BOXPLOT orientation fix (one box per TYPE)
    # ---------------------------------------------------------
    ax.boxplot(
        data.T,
        patch_artist=True,
        showfliers=False,
        whis=[0, 100],
        boxprops=dict(linewidth=1.5, facecolor='none', edgecolor='black'),
        whiskerprops=dict(linewidth=1.5, color='black'),
        capprops=dict(linewidth=1.5, color='black'),
        medianprops=dict(linewidth=2, color='black')
    )

    # ---------------------------------------------------------
    #  2. SCATTER (JITTERED) in fixed correlation order
    # ---------------------------------------------------------
    jitter_width = 0.15
    x_positions = np.arange(1, len(types) + 1)

    for i, corr in enumerate(corr_cols):
        y_vals = data[:, i]

        x_jitter = x_positions + np.random.uniform(
            low=-jitter_width, high=jitter_width, size=len(y_vals)
        )

        ax.scatter(
            x_jitter,
            y_vals,
            marker=marker_styles[i],
            color=correlation_map[corr]["color"],
            edgecolor="black",
            s=90,
            linewidth=0.6,
            alpha=0.85
        )

    # ---------------------------------------------------------
    #  3. LEGEND (fixed order, clean)
    # ---------------------------------------------------------
    legend_elements = [
        Line2D(
            [0], [0],
            marker=marker_styles[i],
            color=correlation_map[corr]["color"],
            linestyle='None',
            markersize=10,
            markeredgecolor='black',
            linewidth=0.6,
            label=correlation_map[corr]["legend"]
        )
        for i, corr in enumerate(corr_cols)
    ]

    ax.legend(handles=legend_elements, title="Correlation",
              frameon=False, fontsize=12)

    # ---------------------------------------------------------
    #  4. LABELS, GRID & EXPORT
    # ---------------------------------------------------------
    formatted_labels = [t.replace("+", "\n+") for t in types]

    ax.set_xticks(x_positions)
    ax.set_xticklabels(formatted_labels, fontsize=16)
    ax.set_ylabel("Potential savings in LCOW (%)", fontsize=16)
    ax.set_ylim(0, 40)
    plt.yticks(fontsize=16)

    # grid in the back with low alpha and ticks
    ax.set_axisbelow(True)

    # keep only major ticks on x-axis (enable minor ticks only for y)
    ax.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

    ax.tick_params(axis='both', which='major', length=6, width=1)
    ax.tick_params(axis='y', which='minor', length=3, width=0.8)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False, length=0)

    fig.tight_layout()
    fig.savefig(f"{sheet}_box_plot.svg", format="svg", dpi=300)
    plt.close(fig)
