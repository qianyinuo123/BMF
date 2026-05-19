import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_simulation_comparison():
    """
    Compare simulation results for two scenarios:
    - All contacts (baseline)
    - No work (intervention)
    """
    countries = ["Uganda", "Qatar", "Monaco", "Germany"]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    plt.figure(figsize=(8, 6))

    for i, country in enumerate(countries):
        # Read all contacts scenario data (All contacts)
        all_file = f"Gillespie_SIR_{country}_all.csv"
        if os.path.exists(all_file):
            df_all = pd.read_csv(all_file)
            beta_all = df_all['Beta']
            mean_all = df_all['mean_infected_ratio']
            ci95_low_all = df_all['ci95_low']
            ci95_high_all = df_all['ci95_high']

            # Scatter points (All contacts)
            plt.scatter(beta_all, mean_all,
                        color=colors[i], marker=markers[i], s=30,
                        label=f'{country} (Sim-All)', zorder=5)

            # Add 95% confidence interval error bars
            y_low = mean_all - ci95_low_all
            y_high = ci95_high_all - mean_all
            y_low = np.maximum(y_low, 0)
            y_high = np.maximum(y_high, 0)

            plt.errorbar(beta_all, mean_all,
                         yerr=[y_low, y_high],
                         fmt='none',
                         ecolor=colors[i],
                         alpha=0.5,
                         capsize=3,
                         capthick=1.5,
                         elinewidth=1.2)

        # Read no work scenario data (No work)
        no_work_file = f"Gillespie_SIR_{country}.csv"
        if os.path.exists(no_work_file):
            df_no_work = pd.read_csv(no_work_file)
            beta_no_work = df_no_work['Beta']
            mean_no_work = df_no_work['mean_infected_ratio']
            ci95_low_no_work = df_no_work['ci95_low']
            ci95_high_no_work = df_no_work['ci95_high']

            # Solid line (No work)
            plt.plot(beta_no_work, mean_no_work,
                     color=colors[i], linewidth=2,
                     label=f'{country} (Sim-No_work)', linestyle='-')

            # Add error bars (No work)
            y_low = mean_no_work - ci95_low_no_work
            y_high = ci95_high_no_work - mean_no_work
            y_low = np.maximum(y_low, 0)
            y_high = np.maximum(y_high, 0)

            plt.errorbar(beta_no_work, mean_no_work,
                         yerr=[y_low, y_high],
                         fmt='none',
                         ecolor=colors[i],
                         alpha=0.3,
                         capsize=2,
                         capthick=1,
                         elinewidth=1)

    plt.xlabel(r'$\beta$', fontsize=20)
    plt.ylabel('Final epidemic size', fontsize=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()

    plt.savefig('simulation_comparison_all_vs_no_work.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_simulation_comparison()