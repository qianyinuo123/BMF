import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def aggregate_theory_to_4_age_groups(df_theory):
    """
    Aggregate theory data from 16 age groups (R1-R16) to 4 coarse age groups
    Age group mapping:
    - 0-19: R1-R4 (age groups 1-4)
    - 20-39: R5-R8 (age groups 5-8)
    - 40-59: R9-R12 (age groups 9-12)
    - 60+: R13-R16 (age groups 13-16)
    """
    df_result = df_theory[['Beta']].copy()

    # 0-19: R1 to R4
    df_result['0-19_mean'] = df_theory[['R1', 'R2', 'R3', 'R4']].mean(axis=1)

    # 20-39: R5 to R8
    df_result['20-39_mean'] = df_theory[['R5', 'R6', 'R7', 'R8']].mean(axis=1)

    # 40-59: R9 to R12
    df_result['40-59_mean'] = df_theory[['R9', 'R10', 'R11', 'R12']].mean(axis=1)

    # 60+: R13 to R16
    df_result['60+_mean'] = df_theory[['R13', 'R14', 'R15', 'R16']].mean(axis=1)

    return df_result


def plot_age_group_comparison_two_simulations():
    """
    Compare theory vs simulation results:
    - Poison_beta_{country}.csv                 → solid lines (theory data, aggregated to 4 coarse age groups)
    - Gillespie_4AgeGroups_{country}.csv        → scatter points + error bars (simulation data)
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
            theory_file = f"Poison_beta_{country}.csv"
            sim_file = f"Gillespie_4AgeGroups_{country}.csv"

            if not os.path.exists(theory_file):
                print(f"File not found: {theory_file}")
                continue
            if not os.path.exists(sim_file):
                print(f"File not found: {sim_file}")
                continue

            df_theory_raw = pd.read_csv(theory_file)
            df_sim = pd.read_csv(sim_file)

            # Aggregate theory data to 4 coarse age groups
            df_theory = aggregate_theory_to_4_age_groups(df_theory_raw)

            beta_theory = df_theory['Beta']
            beta_sim = df_sim['Beta']

            # Get column names for the current age group
            mean_col = f'{age_group}_mean'
            ci_low_col = f'{age_group}_ci95_low'
            ci_high_col = f'{age_group}_ci95_high'

            if mean_col not in df_sim.columns:
                print(f"{country} - {sim_file} missing column: {mean_col}")
                continue

            # ===== Solid line: Theory data =====
            theory_col = f'{age_group}_mean'
            if theory_col in df_theory.columns:
                ax.plot(
                    beta_theory,
                    df_theory[theory_col],
                    color=colors[country_idx],
                    linewidth=2.5,
                    linestyle='-',
                    alpha=0.8,
                    label=f'{country} (Theory)' if age_idx == 0 else ''
                )
            else:
                print(f"{country} theory data missing column: {theory_col}")

            # ===== Scatter points: Simulation data (mean + error bars) =====
            ax.scatter(
                beta_sim,
                df_sim[mean_col],
                color=colors[country_idx],
                marker=markers[country_idx],
                s=50,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.8,
                zorder=5,
                label=f'{country} (Simulation)' if age_idx == 0 else ''
            )

            # Add error bars for simulation (95% CI)
            if ci_low_col in df_sim.columns and ci_high_col in df_sim.columns:
                y_low = df_sim[mean_col] - df_sim[ci_low_col]
                y_high = df_sim[ci_high_col] - df_sim[mean_col]
                y_low = np.maximum(y_low, 0)
                y_high = np.maximum(y_high, 0)

                ax.errorbar(
                    beta_sim,
                    df_sim[mean_col],
                    yerr=[y_low, y_high],
                    fmt='none',
                    ecolor=colors[country_idx],
                    alpha=0.4,
                    capsize=3,
                    capthick=1.2,
                    elinewidth=1.2
                )

            # ===== Subplot formatting =====
            ax.set_title(f'{age_label}', fontsize=18, fontweight='bold', pad=12)
            ax.set_xlabel(r'$\beta$', fontsize=16)
            ax.set_ylabel('Final epidemic size', fontsize=16)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=14)

            # Set Y axis range
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, max(beta_sim.max(), beta_theory.max()))

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
        # Solid line legend (theory)
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                color=colors[i],
                linewidth=3,
                label=f'{country} (Theory)'
            )
        )
        # Scatter legend (simulation)
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                marker=markers[i],
                color='w',
                markerfacecolor=colors[i],
                markersize=10,
                markeredgecolor='black',
                label=f'{country} (Simulation)'
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
    Plot 4 age groups comparison (theory vs simulation) for a single country
    """
    age_groups = ['0-19', '20-39', '40-59', '60+']
    age_group_labels = ['Age 0-19', 'Age 20-39', 'Age 40-59', 'Age 60+']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    theory_file = f"Poison_beta_{country_name}.csv"
    sim_file = f"Gillespie_4AgeGroups_{country_name}.csv"

    if not os.path.exists(theory_file):
        print(f"File not found: {theory_file}")
        return
    if not os.path.exists(sim_file):
        print(f"File not found: {sim_file}")
        return

    df_theory_raw = pd.read_csv(theory_file)
    df_sim = pd.read_csv(sim_file)
    df_theory = aggregate_theory_to_4_age_groups(df_theory_raw)

    color_theory = '#2E86AB'   # Blue (theory)
    color_sim = '#E74C3C'      # Red (simulation)

    for idx, (age_group, age_label) in enumerate(zip(age_groups, age_group_labels)):
        ax = axes[idx]

        mean_col = f'{age_group}_mean'
        ci_low_col = f'{age_group}_ci95_low'
        ci_high_col = f'{age_group}_ci95_high'
        theory_col = f'{age_group}_mean'

        # ===== Solid line: Theory data =====
        ax.plot(
            df_theory['Beta'],
            df_theory[theory_col],
            color=color_theory,
            linewidth=2.5,
            linestyle='-',
            label='Theory'
        )

        # ===== Scatter points: Simulation data (mean + error bars) =====
        ax.scatter(
            df_sim['Beta'],
            df_sim[mean_col],
            color=color_sim,
            marker='o',
            s=40,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.8,
            label='Simulation'
        )

        # Add error bars for simulation (95% CI)
        if ci_low_col in df_sim.columns and ci_high_col in df_sim.columns:
            y_low = df_sim[mean_col] - df_sim[ci_low_col]
            y_high = df_sim[ci_high_col] - df_sim[mean_col]
            y_low = np.maximum(y_low, 0)
            y_high = np.maximum(y_high, 0)

            ax.errorbar(
                df_sim['Beta'],
                df_sim[mean_col],
                yerr=[y_low, y_high],
                fmt='none',
                ecolor=color_sim,
                alpha=0.4,
                capsize=3,
                capthick=1.2,
                elinewidth=1.2
            )

        ax.set_title(f'{age_label}', fontsize=16, fontweight='bold')
        ax.set_xlabel(r'$\beta$', fontsize=14)
        ax.set_ylabel('Final epidemic size', fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='best', fontsize=11)

    plt.suptitle(f'{country_name}: Theory vs Simulation (4 Age Groups)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = f'{country_name}_theory_vs_simulation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Generated: {output_file}")


def check_files_exist():
    """Check if all required files exist"""
    countries = ["Germany", "Uganda", "Qatar", "Monaco"]

    print("\nChecking files:")
    print("-" * 70)

    for country in countries:
        theory_file = f"Poison_beta_{country}.csv"
        sim_file = f"Gillespie_4AgeGroups_{country}.csv"

        theory_exists = os.path.exists(theory_file)
        sim_exists = os.path.exists(sim_file)

        status = "✓" if (theory_exists and sim_exists) else "✗"
        print(f"{status} {country}:")
        print(f"     Theory: {theory_file} ({theory_exists})")
        print(f"     Simulation: {sim_file} ({sim_exists})")

    print("-" * 70)


if __name__ == "__main__":
    # Check files first
    check_files_exist()

    # Plot 4 age groups comparison (all countries)
    plot_age_group_comparison_two_simulations()

    # Optional: Plot single country detailed 4 age groups
    # plot_single_country_four_groups("Germany")