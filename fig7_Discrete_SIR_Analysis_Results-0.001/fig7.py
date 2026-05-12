import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_theory_simulation_comparison():
    countries = ["Uganda", "Qatar", "Monaco", "Germany"]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    plt.figure(figsize=(8, 6))

    for i, country in enumerate(countries):
        theory_file = f"Poison_beta_{country}.csv"
        if os.path.exists(theory_file):
            theory_df = pd.read_csv(theory_file)
            theory_beta = theory_df['Beta']
            theory_r_all = theory_df['R_all']

            plt.plot(theory_beta, theory_r_all,
                     color=colors[i], linewidth=2, label=f'{country} (Theory)', linestyle='-')

        sim_file = f"Discrete_SIR_{country}.csv"
        if os.path.exists(sim_file):
            sim_df = pd.read_csv(sim_file)
            sim_beta = sim_df['Beta']
            sim_mean_infected = sim_df['mean_infected_ratio']

            plt.scatter(sim_beta, sim_mean_infected,
                        color=colors[i], marker=markers[i], s=30,
                        label=f'{country} (Simulation)', zorder=5)

    plt.xlabel(r'$\beta$', fontsize=20)
    plt.ylabel('Final epidemic size', fontsize=20)
    #plt.title('Comparison of Theoretical vs Simulation Results\nfor Different Countries', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()

    plt.savefig('theory_simulation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_theory_simulation_comparison()
