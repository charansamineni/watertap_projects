import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from watertap_spacer_value.plotting_tools.paul_tol_color_maps import gen_paultol_colormap
from watertap_spacer_value.analysis.spacer_optimization_paper.data_analysis.cost_savings_opportunities.overall_cost_savings_opportunities import set_plot_style

def process_design_data():
    for design in ["system", "spacer", "module"]:
        if design == "system":
            n_samples = 7
            design_var_1 = "modules_per_pv"
            base_design_var_1 = 6
            design_var_2 = "n_pv"
            base_design_var_2 = 180
        elif design == "spacer":
            n_samples = 11
            design_var_1 = "sh_multiplier"
            base_design_var_1 = 1
            design_var_2 = "friction_factor_multiplier"
            base_design_var_2 = 1
        elif design == "module":
            n_samples = 11
            design_var_1 = "channel_height"
            base_design_var_1 = 28 * 2.54e-5  # Convert mil to meters
            design_var_2 = "spacer_porosity"
            base_design_var_2 = 0.85
        else:
            raise ValueError("design must be either 'system' or 'spacer'")

        overall_savings = []
        for correlation in ["guillen", "schock", "dacosta", "koustou", "kuroda"]:
            file = f"../../sensitivity_analysis/SWRO_nfe10_vel0.2_sal35_{correlation}_samples{n_samples}_design{design}"
            file += "/interpolated_sweep_results.csv"
            df = pd.read_csv(file)
            lcow = df["LCOW"]
            if lcow.any() in [np.nan, None]:
                raise ValueError(f"LCOW contains NaN or None values for {correlation}")

            base_mask = (df[design_var_1] == base_design_var_1) & (df[design_var_2] == base_design_var_2)
            if not base_mask.any():
                raise ValueError(f"No base case found for {correlation}")
            base_lcow = lcow.loc[base_mask].iloc[0]

            pct_change_lcow = (lcow - base_lcow) / base_lcow * 100

            # Save the dataframe with percentage change
            savings_df = pd.DataFrame({
                "correlation": correlation,
                f'{design_var_1}': df[design_var_1],
                f'{design_var_2}': df[design_var_2],
                "LCOW_pct_change": pct_change_lcow,
                f'{design}': design,
            })
            overall_savings.append(savings_df)

        final_df = pd.concat(overall_savings, ignore_index=True)
        avg_df = (
            final_df
            .groupby(
                [f'{design_var_1}', f'{design_var_2}'],
                as_index=False
            )["LCOW_pct_change"]
            .mean()
            .rename(columns={"LCOW_pct_change": "LCOW_pct_change_avg"})
        )
        avg_df.head()
        avg_df.to_csv(f"LCOW_pct_change_avg_across_correlations_{design}.csv", index=False)


def plot_average_savings_across_designs():
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=False)
    set_plot_style()
    # Colormap
    colors, mappable = gen_paultol_colormap("tol_BuRd", num_samples=5, vmin=0, vmax=1, return_map=True)
    cmap = mappable.cmap

    designs = ["spacer", "module", "system"]

    for ax, design in zip(axes, designs):
        ax.set_box_aspect(1)
        df = pd.read_csv(f"LCOW_pct_change_avg_across_correlations_{design}.csv")

        if design == "system":
            x_var = "modules_per_pv"
            y_var = "n_pv"
            x_label = "RO modules per vessel"
            y_label = "Pressure vessels (#)"
        elif design == "spacer":
            x_var = "sh_multiplier"
            y_var = "friction_factor_multiplier"
            x_label = "Change in Sherwood number (%)"
            y_label = "Change in friction factor (%)"
            df[x_var] = (df[x_var] - 1) * 100
            df[y_var] = (df[y_var] - 1) * 100
        elif design == "module":
            x_var = "channel_height"
            y_var = "spacer_porosity"
            x_label = "Channel height (mil)"
            y_label = "Spacer porosity (#)"

        pivot_table = df.pivot(index=y_var, columns=x_var, values="LCOW_pct_change_avg")
        X = pivot_table.columns.values if design != "module" else pivot_table.columns.values * 1/2.54e-5
        Y = pivot_table.index.values
        Z = pivot_table.values

        # Color normalization
        if design == "system":
            vmin, vmax = -60, 60
        else:
            vmin, vmax = -10, 10
        norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)

        c = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap, norm=norm)

        ax.set_title(f"{design.capitalize()} scale", fontsize=12, fontname="Arial")
        ax.set_xlabel(x_label, fontsize=12, fontname="Arial")
        ax.set_ylabel(y_label, fontsize=12, fontname="Arial")

        # X and Y ticks
        if design == "system":
            ax.set_xticks(np.linspace(X.min(), X.max(), 7))
            ax.set_yticks(np.linspace(Y.min(), Y.max(), 7))
        else:
            ax.set_xticks(np.linspace(X.min(), X.max(), 11))
            ax.set_yticks(np.linspace(Y.min(), Y.max(), 11))

        ax.tick_params(axis='x', labelsize=12)
        for lbl in ax.get_xticklabels():
            lbl.set_fontname("Arial")
            lbl.set_rotation(45 if design == "spacer" else 0)
        ax.tick_params(axis='y', labelsize=12)
        for lbl in ax.get_yticklabels():
            lbl.set_fontname("Arial")

        # Colorbar with proper size and font
        cbar = fig.colorbar(
            c,
            ax=ax,
            orientation="vertical",
            pad=0.05,
            fraction=0.046,
            aspect=20,
            ticks=np.linspace(vmin, vmax, 5)
        )

        # Rotate tick labels
        cbar.ax.tick_params(labelsize=12)

        # Rotate colorbar label
        cbar.set_label("Change in LCOW (%)", fontsize=12, fontname="Arial", rotation=270)

    plt.subplots_adjust(wspace=0.4, hspace=0.4, bottom=0.15)
    plt.savefig("pct_change_LCOW_across_designs.svg", format="svg", dpi=300, transparent=True, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_average_savings_across_designs()
