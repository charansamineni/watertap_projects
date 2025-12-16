import numpy as np
from watertap.flowsheets.spacer_value_paper.SWRO.analyze_savings.sh_pn_savings.evaluate_topology_optimization_savings import (
    init_build_model,
    grab_results,
    solve_for_target_recovery,
)
from watertap.core.membrane_channel_base import Correlations
from watertap.unit_models.reverse_osmosis_1D import (
    ReverseOsmosis1D,
)
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
)
import pandas as pd
import itertools


def module_design_sensitivity():
    results_df = []
    list_of_correlations = [
        Correlations.Eric_Hoek,
        Correlations.DaCosta,
        Correlations.Schock_Miquel,
        Correlations.Koustou,
        Correlations.Kuroda,
    ]
    porosities = np.linspace(0.75, 0.95, 11)
    hc_values = np.linspace(23, 33, 11)
    if not np.isclose(0.85, porosities).any() and not np.isclose(28, hc_values).any():
        raise ValueError(
            "Ensure that base case of 0.85 porosity and 28 mil channel height is included"
        )

    for correlation in list_of_correlations:
        print(f"Running for correlation: {correlation.name}")
        m = init_build_model(correlation_type=correlation)
        solve_for_target_recovery(m, target_recovery=0.5)
        assert np.isclose(m.fs.water_recovery.value, 0.5, atol=1e-2)
        assert m.fs.ro_stages[1].feed_side.Sh_improvement_factor.value == 1.0
        assert m.fs.ro_stages[1].feed_side.friction_factor_improvement.value == 1.0

        r = grab_results(m)
        results_df.append(r)

        for p in porosities:
            for hc in hc_values:
                print(
                    f"Running for porosity: {p:.2f}, hc: {hc:.2f} with correlation: {correlation.name}"
                )
                m.fs.ro_stages[1].feed_side.spacer_porosity.fix(p)
                m.fs.ro_stages[1].feed_side.channel_height.fix(hc * 2.54e-5)  # mil to m
                solve_for_target_recovery(m, target_recovery=0.5)
                r = grab_results(m)
                results_df.append(r)

    results_df = pd.DataFrame(results_df)

    results_with_base = results_df.merge(
        results_df[
            (
                results_df["channel_height"] == 0.7112e-3
            )  # Base case channel height in meters
            & (results_df["porosity"] == 0.85)  # Base case porosity
        ][
            [
                "correlation",
                "LCOW",
                "sec",
                "op_pressure",
                "dp_per_m",
                "average_cp",
            ]
        ],
        on="correlation",
        suffixes=("", "_base"),
    )

    # Calculate percent improvements
    results_with_base["LCOW_savings_pct"] = (
        (results_with_base["LCOW_base"] - results_with_base["LCOW"])
        / results_with_base["LCOW_base"]
    ) * 100

    results_with_base["sec_savings_pct"] = (
        (results_with_base["sec_base"] - results_with_base["sec"])
        / results_with_base["sec_base"]
    ) * 100

    results_with_base["pressure_savings_pct"] = (
        (results_with_base["op_pressure_base"] - results_with_base["op_pressure"])
        / results_with_base["op_pressure_base"]
    ) * 100

    results_with_base["dp_per_m_savings_pct"] = (
        (results_with_base["dp_per_m_base"] - results_with_base["dp_per_m"])
        / results_with_base["dp_per_m_base"]
    ) * 100

    results_with_base["average_cp_savings_pct"] = (
        (results_with_base["average_cp_base"] - results_with_base["average_cp"])
        / results_with_base["average_cp_base"]
    ) * 100

    summary_stats = (
        results_with_base.groupby(["porosity", "channel_height"])[
            [
                "LCOW_savings_pct",
                "sec_savings_pct",
                "pressure_savings_pct",
                "dp_per_m_savings_pct",
                "average_cp_savings_pct",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_stats.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in summary_stats.columns.values
    ]

    with pd.ExcelWriter("results_summary.xlsx", engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="All_Results", index=False)
        summary_stats.to_excel(writer, sheet_name="Avg_Improvement", index=False)

        for metric in [
            "LCOW_savings_pct",
            "sec_savings_pct",
            "pressure_savings_pct",
            "dp_per_m_savings_pct",
            "average_cp_savings_pct",
        ]:
            pivot_mean = summary_stats.pivot(
                index="porosity",
                columns="channel_height",
                values=f"{metric}_mean",
            )
            pivot_std = summary_stats.pivot(
                index="porosity", columns="channel_height", values=f"{metric}_std"
            )
            pivot_mean.to_excel(writer, sheet_name=metric.replace("_pct", "") + "_mean")
            pivot_std.to_excel(writer, sheet_name=metric.replace("_pct", "") + "_std")


if __name__ == "__main__":
    module_design_sensitivity()
