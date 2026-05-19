import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_age_group_comparison_two_simulations():
    """
    Compare two simulation scenarios for 4 age groups:
    - Gillespie_4AgeGroups_{country}.csv        → solid lines + error bars
    - Gillespie_4AgeGroups_{country}_all.csv    → scatter points + error bars
    """

    # ================= Countries & Age Groups =================
    countries = ["Germany", "Uganda", "Qatar", "Monaco"]
    # 4 coarse age groups
    age_groups = ['0-19', '20-39', '40-59', '60+']
    age_group_labels = ['Age 0-19', 'Age 20-39', 'Age 40-59', 'Age 60+']

    # Colors & markers (different colors and shapes for different countries)
    colors = ['#8AB4D6', '#FFB87A', '#8CCB8C', '#FF7F7F']
    markers = ['o', 's', '^', 'D']

    # ================= Create subplot layout: 2 rows x 2 columns (4 age groups) =================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # ================= Loop over age groups =================
    for age_idx, (age_group, age_label) in enumerate(zip(age_groups, age_group_labels)):
        ax = axes[age_idx]

        # ================= Loop over countries =================
        for country_idx, country in enumerate(countries):
            print(f"Processing {country} - {age_group}...")

            # File names
            sim_line_file = f"Gillespie_4AgeGroups_{country}.csv"
            sim_scatter_file = f"Gillespie_4AgeGroups_{country}_all.csv"

            if not os.path.exists(sim_line_file):
                print(f"File not found: {sim_line_file}")
                continue
            if not os.path.exists(sim_scatter_file):
                print(f"File not found: {sim_scatter_file}")
                continue

            df_line = pd.read_csv(sim_line_file)
            df_scatter = pd.read_csv(sim_scatter_file)

            beta_line = df_line['Beta']
            beta_scatter = df_scatter['Beta']

            # Get column names for the current age group
            mean_col = f'{age_group}_mean'
            ci_low_col = f'{age_group}_ci95_low'
            ci_high_col = f'{age_group}_ci95_high'

            if mean_col not in df_line.columns:
                print(f"{country} - {sim_line_file} missing column: {mean_col}")
                continue
            if mean_col not in df_scatter.columns:
                print(f"{country} - {sim_scatter_file} missing column: {mean_col}")
                continue

            # ===== Solid line: No_school scenario (mean + error bars) =====
            ax.plot(
                beta_line,
                df_line[mean_col],
                color=colors[country_idx],
                linewidth=2.5,
                linestyle='-',
                alpha=0.8,
                label=f'{country} (Sim-No_school)' if age_idx == 0 else ''
            )

            # Add error bars for solid line (confidence interval)
            if ci_low_col in df_line.columns and ci_high_col in df_line.columns:
                step = max(1, len(beta_line) // 30)
                y_low = df_line[mean_col][::step] - df_line[ci_low_col][::step]
                y_high = df_line[ci_high_col][::step] - df_line[mean_col][::step]
                y_low = np.maximum(y_low, 0)
                y_high = np.maximum(y_high, 0)

                ax.errorbar(
                    beta_line[::step],
                    df_line[mean_col][::step],
                    yerr=[y_low, y_high],
                    fmt='none',
                    ecolor=colors[country_idx],
                    alpha=0.4,
                    capsize=3,
                    capthick=1.2,
                    elinewidth=1.2
                )

            # ===== Scatter points: All scenario (mean + error bars) =====
            ax.scatter(
                beta_scatter,
                df_scatter[mean_col],
                color=colors[country_idx],
                marker=markers[country_idx],
                s=50,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.8,
                zorder=5,
                label=f'{country} (Sim-All)' if age_idx == 0 else ''
            )

            # Add error bars for scatter points (confidence interval)
            if ci_low_col in df_scatter.columns and ci_high_col in df_scatter.columns:
                step = max(1, len(beta_scatter) // 20)
                y_low = df_scatter[mean_col][::step] - df_scatter[ci_low_col][::step]
                y_high = df_scatter[ci_high_col][::step] - df_scatter[mean_col][::step]
                y_low = np.maximum(y_low, 0)
                y_high = np.maximum(y_high, 0)

                ax.errorbar(
                    beta_scatter[::step],
                    df_scatter[mean_col][::step],
                    yerr=[y_low, y_high],
                    fmt='none',
                    ecolor=colors[country_idx],
                    alpha=0.3,
                    capsize=2,
                    capthick=1,
                    elinewidth=1
                )

            # ===== Subplot formatting =====
            ax.set_title(f'{age_label}', fontsize=18, fontweight='bold', pad=12)
            ax.set_xlabel(r'$\beta$', fontsize=16)
            ax.set_ylabel('Final epidemic size', fontsize=16)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=14)

            # Set Y axis range
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, df_line['Beta'].max())

    # ================= Add legend =================
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=14,
               frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Save figure
    output_file = 'age_group_four_groups_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"Generated: {output_file}")

    # ================= Save legend separately =================
    legend_fig, legend_ax = plt.subplots(figsize=(12, 2.5))
    legend_ax.axis('off')

    legend_elements = []
    for i, country in enumerate(countries):
        # Solid line legend
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                color=colors[i],
                linewidth=3,
                label=f'{country} (Sim-No_school)'
            )
        )
        # Scatter legend
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                marker=markers[i],
                color='w',
                markerfacecolor=colors[i],
                markersize=10,
                markeredgecolor='black',
                label=f'{country} (Sim-All)'
            )
        )
        # Error bar legend (add only once)
        if i == 0:
            legend_elements.append(
                plt.Line2D(
                    [0], [0],
                    color='gray',
                    linewidth=1.5,
                    alpha=0.5,
                    label='95% CI'
                )
            )

    legend_ax.legend(
        handles=legend_elements,
        loc='center',
        ncol=3,
        fontsize=12,
        frameon=True,
        fancybox=True
    )

    legend_fig.savefig('age_group_comparison_legend.png', dpi=300, bbox_inches='tight')
    plt.close(legend_fig)
    print("Generated: age_group_comparison_legend.png")


def plot_single_country_four_groups(country_name):
    """
    Plot 4 age groups comparison for a single country
    """
    age_groups = ['0-19', '20-39', '40-59', '60+']
    age_group_labels = ['Age 0-19', 'Age 20-39', 'Age 40-59', 'Age 60+']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    sim_line_file = f"Gillespie_4AgeGroups_{country_name}.csv"
    sim_scatter_file = f"Gillespie_4AgeGroups_{country_name}_all.csv"

    if not os.path.exists(sim_line_file):
        print(f"File not found: {sim_line_file}")
        return
    if not os.path.exists(sim_scatter_file):
        print(f"File not found: {sim_scatter_file}")
        return

    df_line = pd.read_csv(sim_line_file)
    df_scatter = pd.read_csv(sim_scatter_file)

    color_line = '#2E86AB'
    color_scatter = '#E74C3C'

    for idx, (age_group, age_label) in enumerate(zip(age_groups, age_group_labels)):
        ax = axes[idx]

        mean_col = f'{age_group}_mean'
        ci_low_col = f'{age_group}_ci95_low'
        ci_high_col = f'{age_group}_ci95_high'

        # ===== Solid line: No_school (mean + error bars) =====
        ax.plot(
            df_line['Beta'],
            df_line[mean_col],
            color=color_line,
            linewidth=2.5,
            linestyle='-',
            label='No_school contacts'
        )

        # Add error bars for solid line (confidence interval)
        if ci_low_col in df_line.columns and ci_high_col in df_line.columns:
            step = max(1, len(df_line['Beta']) // 30)
            y_low = df_line[mean_col][::step] - df_line[ci_low_col][::step]
            y_high = df_line[ci_high_col][::step] - df_line[mean_col][::step]
            y_low = np.maximum(y_low, 0)
            y_high = np.maximum(y_high, 0)

            ax.errorbar(
                df_line['Beta'][::step],
                df_line[mean_col][::step],
                yerr=[y_low, y_high],
                fmt='none',
                ecolor=color_line,
                alpha=0.4,
                capsize=3,
                capthick=1.2,
                elinewidth=1.2
            )

        # ===== Scatter points: All (mean + error bars) =====
        ax.scatter(
            df_scatter['Beta'],
            df_scatter[mean_col],
            color=color_scatter,
            marker='o',
            s=40,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.8,
            label='All contacts'
        )

        # Add error bars for scatter points (confidence interval)
        if ci_low_col in df_scatter.columns and ci_high_col in df_scatter.columns:
            step = max(1, len(df_scatter['Beta']) // 30)
            y_low = df_scatter[mean_col][::step] - df_scatter[ci_low_col][::step]
            y_high = df_scatter[ci_high_col][::step] - df_scatter[mean_col][::step]
            y_low = np.maximum(y_low, 0)
            y_high = np.maximum(y_high, 0)

            ax.errorbar(
                df_scatter['Beta'][::step],
                df_scatter[mean_col][::step],
                yerr=[y_low, y_high],
                fmt='none',
                ecolor=color_scatter,
                alpha=0.3,
                capsize=2
            )

        ax.set_title(f'{age_label}', fontsize=16, fontweight='bold')
        ax.set_xlabel(r'$\beta$', fontsize=14)
        ax.set_ylabel('Final epidemic size', fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='best', fontsize=11)

    plt.tight_layout()

    output_file = f'{country_name}_four_groups_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Generated: {output_file}")


def check_files_exist():
    """Check if all required files exist"""
    countries = ["Germany", "Uganda", "Qatar", "Monaco"]

    print("\nChecking files:")
    print("-" * 70)

    for country in countries:
        line_file = f"Gillespie_4AgeGroups_{country}.csv"
        all_file = f"Gillespie_4AgeGroups_{country}_all.csv"

        line_exists = os.path.exists(line_file)
        all_exists = os.path.exists(all_file)

        status = "✓" if (line_exists and all_exists) else "✗"
        print(f"{status} {country}:")
        print(f"     {line_file} ({line_exists})")
        print(f"     {all_file} ({all_exists})")

    print("-" * 70)


if __name__ == "__main__":
    # Check files first
    check_files_exist()

    # Plot 4 age groups comparison (all countries)
    plot_age_group_comparison_two_simulations()

    # Optional: Plot single country detailed 4 age groups
    # plot_single_country_four_groups("Uganda")