from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_lcow_sec_by_case_study(output_path="lcow_sec_subplots.png", df=None):

    case_studies = df['Case Study'].unique()
    spacers = df['Spacer'].unique()
    lcow = np.array([
        [df[(df['Case Study'] == cs) & (df['Spacer'] == sp)]['LCOW'].values[0] for sp in spacers]
        for cs in case_studies
    ])
    sec = np.array([
        [df[(df['Case Study'] == cs) & (df['Spacer'] == sp)]['SEC'].values[0] for sp in spacers]
        for cs in case_studies
    ])

    cmap = plt.get_cmap("tab10")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    width = 0.2
    x = np.arange(len(case_studies))
    spacer_colors = [cmap(i / (len(spacers) - 1)) for i in range(len(spacers))]

    for i, (metric, ax, data) in enumerate(zip(["LCOW", "SEC"], axes, [lcow, sec])):
        for k, spacer in enumerate(spacers):
            values = data[:, k]
            ax.bar(x + k * width, values, width, label=spacer if i == 0 else "", color=spacer_colors[k], edgecolor="black")
        ax.set_xticks(x + width)
        ax.set_xticklabels(case_studies)
        ax.set_ylabel("LCOW ($/m³)" if metric == "LCOW" else "SEC (kWh/m³)")

        if i == 0:
            ax.legend(loc="best")

        if metric == "LCOW":
            ax.set_ylim(0, 1)
            ax.set_yticks(np.linspace(0, 1, 11))
        else:
            ax.set_ylim(0, 5)
            ax.set_yticks(np.linspace(0, 5, 11))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()




def plot_potential_savings(folder_path, ax=None):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2,
        "legend.frameon": False,
    })

    all_savings = []
    case_study_names = []
    optimization_strategies = None

    for file in os.listdir(folder_path):
        if file.endswith('.xlsx'):
            excel_path = os.path.join(folder_path, file)
            df = pd.read_excel(excel_path, sheet_name='Macro Metrics')
            # Exclude Simulation row if present
            if 'Strategy' in df.columns:
                mask = ~df['Strategy'].str.lower().isin(['simulation', 'spacer'])
                filtered_df = df[mask]
            else:
                filtered_df = df
            case_study_names.append(file.replace('.xlsx', '').split('_')[-1].upper())
            all_savings.append(filtered_df['Savings_vs_Simulation_%'].values)
            if optimization_strategies is None:
                optimization_strategies = filtered_df['Strategy'].values

    # Convert to numpy array for plotting
    savings_arr = np.array(all_savings)
    n_case_studies, n_bars = savings_arr.shape
    x = np.arange(n_bars)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    width = 0.5 / n_case_studies  # Set bar width
    colors = ['#7776BC', '#FF674D']
    for i in range(n_case_studies):
        ax.bar(x + i * width, savings_arr[i], width, label=case_study_names[i], color=colors[i % len(colors)])

    ax.set_ylabel('% Savings')
    ax.set_title('Potential savings by optimization strategy')
    ax.set_xticks(x + width * (n_case_studies - 1) / 2)
    ax.set_xticklabels(optimization_strategies)
    ax.set_ylim([0, 25])
    ax.set_yticks(np.linspace(0, 25, 6))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "potential_savings.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    folder = os.getcwd()
    plot_potential_savings(folder_path=folder)
    # Grab the values for LCOW and SEC from the table in the from the excel files

    # spacer_values = []
    # for file in os.listdir(folder):
    #     if file.endswith('.xlsx'):
    #         excel_path = os.path.join(folder, file)
    #         df_excel = pd.read_excel(excel_path, sheet_name='Macro Metrics')
    #         # Filter for 'Simulation' strategy and '3d-printed spacer (CFD)'
    #         mask = (
    #             (df_excel['Strategy'].str.lower() == 'simulation')
    #         )
    #         filtered = df_excel[mask]
    #         if not filtered.empty:
    #             lcow = filtered['LCOW'].values[0] if 'LCOW' in filtered else None
    #             sec = filtered['Energy Consumption'].values[0] if 'Energy Consumption' in filtered else None
    #             spacer_values.append({
    #                 'File': file,
    #                 'LCOW': lcow,
    #                 'SEC': sec
    #             })
    # print("3d-printed spacer values from Simulation case:")
    # for val in spacer_values:
    #     print(val)


    data = {
        'Case Study': ['SWRO', 'SWRO', 'BWRO', 'BWRO'],
        'Spacer': ['3d-printed spacer (CFD)', 'Conventional spacer (Schock et al.)'] * 2,
        'LCOW': [0.783945455245091, 0.769521024, 0.2012119140729478, 0.194503265],
        'SEC': [4.451530436448317, 4.412580228, 1.02925075738261, 1.108505078]
    }
    df = pd.DataFrame(data)
    print(df)
    # Calculate and print the difference between spacers for each case study
    for cs in df['Case Study'].unique():
        group = df[df['Case Study'] == cs]
        if len(group) == 2:
            if group.iloc[1]['Spacer'].lower().startswith('conventional'):
                base_idx, comp_idx = 1, 0
            else:
                base_idx, comp_idx = 0, 1
            lcow_pct_change = 100 * (group.iloc[comp_idx]['LCOW'] - group.iloc[base_idx]['LCOW']) / \
                              group.iloc[base_idx]['LCOW']
            sec_pct_change = 100 * (group.iloc[comp_idx]['SEC'] - group.iloc[base_idx]['SEC']) / group.iloc[base_idx][
                'SEC']
            print(f"{cs}: LCOW % change = {lcow_pct_change:.2f}%, SEC % change = {sec_pct_change:.2f}%")
    plot_lcow_sec_by_case_study(df=df)

