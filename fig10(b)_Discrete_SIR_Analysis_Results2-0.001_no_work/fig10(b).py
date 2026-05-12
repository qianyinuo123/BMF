import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_simulation_comparison_two_schemes():

    countries = ["Uganda", "Qatar", "Monaco", "Germany"]

    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    plt.figure(figsize=(8, 6))

    for i, country in enumerate(countries):

        sim_file = f"Discrete_SIR_{country}.csv"
        if os.path.exists(sim_file):
            df = pd.read_csv(sim_file)

            plt.plot(
                df['Beta'],
                df['mean_infected_ratio'],
                color=colors[i],
                linewidth=2,
                linestyle='-',
                label=f'{country} (Simulation-No_work)'
            )

        sim_all_file = f"Discrete_SIR_{country}_all.csv"
        if os.path.exists(sim_all_file):
            df_all = pd.read_csv(sim_all_file)

            plt.scatter(
                df_all['Beta'],
                df_all['mean_infected_ratio'],
                color=colors[i],
                marker=markers[i],
                s=30,
                label=f'{country} (Simulation–All)',
                zorder=5
            )

    plt.xlabel(r'$\beta$', fontsize=20)
    plt.ylabel('Final epidemic size', fontsize=20)

    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.savefig(
        'simulation_comparison_two_schemes.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()


if __name__ == "__main__":
    plot_simulation_comparison_two_schemes()
