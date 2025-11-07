import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Custom correlation color and label map
correlation_map = {
    "daCosta": {"color": "#4B0082", "legend": "Da Costa et al."},
    "Guillen": {"color": "#008080", "legend": "Guillen et al."},
    "Koustou": {"color": "#DAA520", "legend": "Koustou et al."},
    "Kuroda": {"color": "#6A5ACD", "legend": "Kuroda et al."},
    "Schock": {"color": "#DC143C", "legend": "Schock et al."},
}

# Raw data
excel_path = "Overall_data.xlsx"
df_swro = pd.read_excel(excel_path, sheet_name="SWRO")
df_bwro = pd.read_excel(excel_path, sheet_name="BWRO")

# Prepare wide dataframes for SWRO and BWRO
df_wide_swro = df_swro
df_wide_bwro = df_bwro

# Reshape to long format
df_long_swro = pd.melt(df_wide_swro, id_vars="Type", var_name="Correlation", value_name="Value")
df_long_bwro = pd.melt(df_wide_bwro, id_vars="Type", var_name="Correlation", value_name="Value")

# Replace long x-axis labels with split format
df_long_swro["Type"] = df_long_swro["Type"].str.replace("+", "\n")
df_long_bwro["Type"] = df_long_bwro["Type"].str.replace("+", "\n")


# Use consistent color mapping
palette = {key: val["color"] for key, val in correlation_map.items()}

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

# SWRO subplot
sns.barplot(
    data=df_long_swro,
    x="Type",
    y="Value",
    hue="Correlation",
    palette=palette,
    hue_order=list(correlation_map.keys()),
    edgecolor="black",
    linewidth=1,
    ax=axs[0]
)
axs[0].set_ylabel("Potential savings (%)", fontsize=14)
axs[0].set_title("A) SWRO", fontsize=16, loc='left', pad=20)
axs[0].set_ylim(0, 40)
axs[0].set_yticks(np.linspace(0, 40, 9))
axs[0].set_xlabel("")
axs[0].tick_params(axis='y', labelsize=14)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].legend_.remove()

# BWRO subplot
sns.barplot(
    data=df_long_bwro,
    x="Type",
    y="Value",
    hue="Correlation",
    palette=palette,
    hue_order=list(correlation_map.keys()),
    edgecolor="black",
    linewidth=1,
    ax=axs[1]
)
axs[1].set_ylabel("Potential savings (%)", fontsize=14)
axs[1].set_title("B) BWRO", fontsize=16, loc='left', pad=20)
axs[1].set_xlabel("Optimization strategy", fontsize=14)
axs[1].set_ylim(0, 40)
axs[1].set_yticks(np.linspace(0, 40, 9))
axs[1].tick_params(axis='y', labelsize=14)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].legend_.remove()

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14
plt.xticks(rotation=0, ha='center')

from matplotlib.patches import Patch

# Create custom legend handles (colored boxes)
legend_handles = [
    Patch(facecolor=correlation_map[correlation]["color"], edgecolor="black", label=correlation_map[correlation]["legend"])
    for correlation in correlation_map.keys()
]

# Place legend below both subplots
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(legend_handles),
    frameon=False,
    fontsize=14,
    bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("overall_savings_plot.svg", dpi=300)
plt.show()
