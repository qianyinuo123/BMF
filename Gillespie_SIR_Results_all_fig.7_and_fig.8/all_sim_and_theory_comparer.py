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
        # 读取理论数据
        theory_file = f"Poison_beta_{country}.csv"
        if os.path.exists(theory_file):
            theory_df = pd.read_csv(theory_file)
            theory_beta = theory_df['Beta']
            theory_r_all = theory_df['R_all']

            plt.plot(theory_beta, theory_r_all,
                     color=colors[i], linewidth=2, label=f'{country} (Theory)', linestyle='-')

        # 读取仿真数据
        sim_file = f"Gillespie_SIR_{country}.csv"
        if os.path.exists(sim_file):
            sim_df = pd.read_csv(sim_file)
            sim_beta = sim_df['Beta']
            sim_mean_infected = sim_df['mean_infected_ratio']
            sim_ci95_low = sim_df['ci95_low']
            sim_ci95_high = sim_df['ci95_high']

            # 绘制散点
            plt.scatter(sim_beta, sim_mean_infected,
                        color=colors[i], marker=markers[i], s=30,
                        label=f'{country} (Simulation)', zorder=5)

            # 添加 95% 置信区间误差条
            y_low = sim_mean_infected - sim_ci95_low
            y_high = sim_ci95_high - sim_mean_infected
            y_low = np.maximum(y_low, 0)
            y_high = np.maximum(y_high, 0)

            plt.errorbar(sim_beta, sim_mean_infected,
                         yerr=[y_low, y_high],
                         fmt='none',
                         ecolor=colors[i],
                         alpha=0.5,
                         capsize=3,
                         capthick=1.5,
                         elinewidth=1.2)

    plt.xlabel(r'$\beta$', fontsize=20)
    plt.ylabel('Final epidemic size', fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()

    plt.savefig('theory_simulation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_theory_simulation_comparison()