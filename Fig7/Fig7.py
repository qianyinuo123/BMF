import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec


def plot_age_group_comparison():
    countries = ["Germany", "Uganda", "Qatar", "Monaco"]


    max_age_groups = 16


    colors = ['#8AB4D6', '#FFB87A', '#8CCB8C', '#FF7F7F']
    markers = ['o', 's', '^', 'D']


    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.25, wspace=0.2)  # 更紧凑的布局


    age_axes = []
    for row in range(4):
        row_axes = []
        for col in range(4):
            age_idx = row * 4 + col
            if age_idx < max_age_groups:
                ax = fig.add_subplot(gs[row, col])
                row_axes.append(ax)
        age_axes.append(row_axes)


    for country_idx, country in enumerate(countries):

        theory_file = f"Poison_beta_{country}.csv"
        theory_data = None
        if os.path.exists(theory_file):
            theory_df = pd.read_csv(theory_file)
            theory_beta = theory_df['Beta']
            theory_data = theory_df


        # 读取仿真数据
        sim_file = f"Discrete_SIR_{country}.csv"
        sim_data = None
        if os.path.exists(sim_file):
            sim_df = pd.read_csv(sim_file)
            sim_beta = sim_df['Beta']
            sim_data = sim_df

        if theory_data is None or sim_data is None:
            continue

        for age_group in range(max_age_groups):
            row = age_group // 4
            col = age_group % 4

            if row < len(age_axes) and col < len(age_axes[row]):
                ax = age_axes[row][col]



                theory_col_name = f'R{age_group + 1}'

                sim_col_name = f'R_age{age_group + 1}'

                if theory_col_name in theory_data.columns and sim_col_name in sim_data.columns:

                    theory_values = theory_data[theory_col_name]
                    ax.plot(theory_beta, theory_values,
                            color=colors[country_idx], linewidth=2,
                            label=f'{country} (Theory)' if age_group == 0 else '',
                            linestyle='-', alpha=0.8)


                    sim_values = sim_data[sim_col_name]
                    ax.scatter(sim_beta, sim_values,
                               color=colors[country_idx], marker=markers[country_idx],
                               s=40, label=f'{country} (Sim)' if age_group == 0 else '',
                               alpha=0.8, edgecolor='white', linewidth=0.5)


                if country_idx == 0:
                    ax.set_title(f'Age Group {age_group}', fontsize=14, fontweight='bold')


                if row == 3:
                    ax.set_xlabel(r'$\beta$', fontsize=12)
                if col == 0:
                    ax.set_ylabel('Final epidemic size', fontsize=12)


                ax.grid(True, alpha=0.3)


                ax.tick_params(axis='both', which='major', labelsize=10)


    plt.suptitle('Comparison of Theoretical vs Simulation Results by Age Group\nfor Germany, Uganda, Qatar, and Monaco',
                 fontsize=16, fontweight='bold', y=0.98)

    legend_fig, legend_ax = plt.subplots(figsize=(10, 2))
    legend_ax.axis('off')


    legend_elements = []
    for i, country in enumerate(countries):

        legend_elements.append(plt.Line2D([0], [0], color=colors[i], linewidth=3,
                                          label=f'{country} (Theory)'))

        legend_elements.append(plt.Line2D([0], [0], marker=markers[i], color='w',
                                          markerfacecolor=colors[i], markersize=10,
                                          label=f'{country} (Simulation)'))


    legend_ax.legend(handles=legend_elements, loc='center', ncol=4,
                     fontsize=14, frameon=True, fancybox=True)

    plt.tight_layout()


    plt.savefig('age_group_comparison_all.png', dpi=300, bbox_inches='tight')
    plt.show()

    legend_fig.savefig('age_group_legend.png', dpi=300, bbox_inches='tight')
    plt.close(legend_fig)



if __name__ == "__main__":

    countries = ["Germany", "Uganda", "Qatar", "Monaco"]
    for country in countries:
        theory_file = f"Poison_beta_{country}.csv"
        sim_file = f"Discrete_SIR_{country}.csv"


    print("=" * 60)

    # 绘制所有年龄组
    plot_age_group_comparison()


    print("=" * 60)
    print("所有图表绘制完成！")
