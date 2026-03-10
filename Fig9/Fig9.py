import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec


def plot_age_group_comparison_two_simulations():

    countries = ["Germany", "Uganda", "Qatar", "Monaco"]
    max_age_groups = 16

    colors  = ['#8AB4D6', '#FFB87A', '#8CCB8C', '#FF7F7F']
    markers = ['o', 's', '^', 'D']


    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.25, wspace=0.2)

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

        sim_line_file    = f"Discrete_SIR_{country}.csv"
        sim_scatter_file = f"Discrete_SIR_{country}_all.csv"

        if not os.path.exists(sim_line_file) or not os.path.exists(sim_scatter_file):

            continue

        df_line    = pd.read_csv(sim_line_file)
        df_scatter = pd.read_csv(sim_scatter_file)

        beta_line    = df_line['Beta']
        beta_scatter = df_scatter['Beta']


        for age_group in range(max_age_groups):
            row = age_group // 4
            col = age_group % 4

            if row >= len(age_axes) or col >= len(age_axes[row]):
                continue

            ax = age_axes[row][col]
            col_name = f'R_age{age_group + 1}'

            if col_name not in df_line.columns or col_name not in df_scatter.columns:
                continue


            ax.plot(
                beta_line,
                df_line[col_name],
                color=colors[country_idx],
                linewidth=2,
                linestyle='-',
                alpha=0.8,
                label=f'{country} (Sim-No_school)' if age_group == 0 else ''
            )


            ax.scatter(
                beta_scatter,
                df_scatter[col_name],
                color=colors[country_idx],
                marker=markers[country_idx],
                s=40,
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5,
                label=f'{country} (Sim–All)' if age_group == 0 else ''
            )

            if country_idx == 0:
                ax.set_title(f'Age Group {age_group}', fontsize=14, fontweight='bold')

            if row == 3:
                ax.set_xlabel(r'$\beta$', fontsize=12)
            if col == 0:
                ax.set_ylabel('Final epidemic size', fontsize=12)

            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=10)


    plt.suptitle(
        'Comparison of Two Simulation Schemes by Age Group\n'
        'Discrete SIR with All Contacts vs Discrete SIR with No_school Contacts',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    legend_fig, legend_ax = plt.subplots(figsize=(10, 2))
    legend_ax.axis('off')

    legend_elements = []
    for i, country in enumerate(countries):
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                color=colors[i],
                linewidth=3,
                label=f'{country} (Sim-No_school)'
            )
        )
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                marker=markers[i],
                color='w',
                markerfacecolor=colors[i],
                markersize=10,
                label=f'{country} (Sim–All)'
            )
        )

    legend_ax.legend(
        handles=legend_elements,
        loc='center',
        ncol=4,
        fontsize=14,
        frameon=True,
        fancybox=True
    )

    plt.tight_layout()
    plt.savefig('age_group_two_simulation_comparison.png', dpi=300, bbox_inches='tight')
    legend_fig.savefig('age_group_two_simulation_legend.png', dpi=300, bbox_inches='tight')

    plt.show()
    plt.close(legend_fig)



if __name__ == "__main__":
    plot_age_group_comparison_two_simulations()
