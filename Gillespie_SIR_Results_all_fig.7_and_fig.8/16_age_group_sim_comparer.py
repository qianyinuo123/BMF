import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_16_age_groups_comparison():
    """
    Compare simulation vs theory results for 16 age groups:
    - Gillespie_SIR_{country}.csv        → scatter points + error bars (simulation data)
    - Poison_beta_{country}.csv          → solid lines (theoretical data R_age{age_group})
    """

    # ================= Countries & Age Groups =================
    countries = ["Germany", "Uganda", "Qatar", "Monaco"]
    # 16 age groups
    age_groups = list(range(1, 17))
    age_group_labels = [f'Age {i}' for i in range(1, 17)]

    # Colors & markers (different colors and shapes for different countries)
    colors = ['#8AB4D6', '#FFB87A', '#8CCB8C', '#FF7F7F']
    markers = ['o', 's', '^', 'D']

    # ================= Create subplot layout: 4 rows x 4 columns (16 age groups) =================
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()

    # ================= Loop over age groups =================
    for age_idx, (age_group, age_label) in enumerate(zip(age_groups, age_group_labels)):
        ax = axes[age_idx]

        # ================= Loop over countries =================
        for country_idx, country in enumerate(countries):
            print(f"Processing {country} - {age_label}...")

            # File name format
            sim_file = f"Gillespie_SIR_{country}.csv"
            theory_file = f"Poison_beta_{country}.csv"

            if not os.path.exists(sim_file):
                print(f"⚠️ File not found: {sim_file}")
                continue
            if not os.path.exists(theory_file):
                print(f"⚠️ File not found: {theory_file}")
                continue

            df_sim = pd.read_csv(sim_file)
            df_theory = pd.read_csv(theory_file)

            beta_sim = df_sim['Beta']
            beta_theory = df_theory['Beta']

            # ===== Simulation data column names =====
            sim_mean_col = f'R_age{age_group}'
            sim_ci_low_col = f'R_age{age_group}_ci95_low'
            sim_ci_high_col = f'R_age{age_group}_ci95_high'

            # ===== Theory data column names =====
            theory_col = f'R{age_group}'  # R1, R2, ..., R16

            if sim_mean_col not in df_sim.columns:
                print(f"⚠️ {country} - {sim_file} missing column: {sim_mean_col}")
                continue
            if theory_col not in df_theory.columns:
                print(f"⚠️ {country} - {theory_file} missing column: {theory_col}")
                continue

            # ===== Solid line: Theory data =====
            ax.plot(
                beta_theory,
                df_theory[theory_col],
                color=colors[country_idx],
                linewidth=2,
                linestyle='-',
                alpha=0.8,
                label=f'{country} (Theory)' if age_idx == 0 else ''
            )

            # ===== Scatter points: Simulation data (mean + error bars) =====
            ax.scatter(
                beta_sim,
                df_sim[sim_mean_col],
                color=colors[country_idx],
                marker=markers[country_idx],
                s=30,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5,
                zorder=5,
                label=f'{country} (Simulation)' if age_idx == 0 else ''
            )

            # Add error bars for simulation (95% CI)
            if sim_ci_low_col in df_sim.columns and sim_ci_high_col in df_sim.columns:
                y_low = df_sim[sim_mean_col] - df_sim[sim_ci_low_col]
                y_high = df_sim[sim_ci_high_col] - df_sim[sim_mean_col]
                y_low = np.maximum(y_low, 0)
                y_high = np.maximum(y_high, 0)

                ax.errorbar(
                    beta_sim,
                    df_sim[sim_mean_col],
                    yerr=[y_low, y_high],
                    fmt='none',
                    ecolor=colors[country_idx],
                    alpha=0.4,
                    capsize=2,
                    capthick=1,
                    elinewidth=1
                )

            # ===== Subplot formatting =====
            ax.set_title(f'{age_label}', fontsize=12, fontweight='bold', pad=8)
            ax.set_xlabel(r'$\beta$', fontsize=10)
            ax.set_ylabel('Final epidemic size', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=9)

            # Set axis ranges
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, max(beta_sim.max(), beta_theory.max()))

    # ================= Add legend =================
    handles, labels = axes[0].get_legend_handles_labels()
    # Remove duplicates
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_handles.append(h)
            unique_labels.append(l)

    fig.legend(unique_handles, unique_labels, loc='lower center', ncol=4, fontsize=11,
               frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)

    # Save figure
    output_file = '16_age_groups_theory_vs_simulation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"✅ Generated: {output_file}")

    # ================= Save legend separately =================
    legend_fig, legend_ax = plt.subplots(figsize=(10, 2.5))
    legend_ax.axis('off')

    legend_elements = []
    for i, country in enumerate(countries):
        # Solid line legend (theory)
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                color=colors[i],
                linewidth=2.5,
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
                markersize=8,
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
        fontsize=11,
        frameon=True,
        fancybox=True
    )

    legend_fig.savefig('16_age_groups_legend.png', dpi=300, bbox_inches='tight')
    plt.close(legend_fig)
    print("✅ Generated: 16_age_groups_legend.png")


def plot_single_country_16_groups(country_name):
    """
    Plot 16 age groups comparison for a single country (4x4 layout)
    Theory: solid lines, Simulation: scatter points + error bars
    """
    age_groups = list(range(1, 17))
    age_group_labels = [f'Age {i}' for i in range(1, 17)]

    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    axes = axes.flatten()

    sim_file = f"Gillespie_SIR_{country_name}.csv"
    theory_file = f"Poison_beta_{country_name}.csv"

    if not os.path.exists(sim_file):
        print(f"❌ File not found: {sim_file}")
        return
    if not os.path.exists(theory_file):
        print(f"❌ File not found: {theory_file}")
        return

    df_sim = pd.read_csv(sim_file)
    df_theory = pd.read_csv(theory_file)

    color_sim = '#2E86AB'  # Blue (simulation)
    color_theory = '#E74C3C'  # Red (theory)

    for idx, (age_group, age_label) in enumerate(zip(age_groups, age_group_labels)):
        ax = axes[idx]

        # ===== Simulation data column names =====
        sim_mean_col = f'R_age{age_group}'
        sim_ci_low_col = f'R_age{age_group}_ci95_low'
        sim_ci_high_col = f'R_age{age_group}_ci95_high'

        # ===== Theory data column names =====
        theory_col = f'R{age_group}'

        if sim_mean_col not in df_sim.columns:
            print(f"⚠️ {sim_file} missing column: {sim_mean_col}")
            continue
        if theory_col not in df_theory.columns:
            print(f"⚠️ {theory_file} missing column: {theory_col}")
            continue

        # ===== Solid line: Theory data =====
        ax.plot(
            df_theory['Beta'],
            df_theory[theory_col],
            color=color_theory,
            linewidth=2,
            linestyle='-',
            label='Theory'
        )

        # ===== Scatter points: Simulation data =====
        ax.scatter(
            df_sim['Beta'],
            df_sim[sim_mean_col],
            color=color_sim,
            marker='o',
            s=25,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            label='Simulation'
        )

        # Add error bars for simulation (95% CI)
        if sim_ci_low_col in df_sim.columns and sim_ci_high_col in df_sim.columns:
            y_low = df_sim[sim_mean_col] - df_sim[sim_ci_low_col]
            y_high = df_sim[sim_ci_high_col] - df_sim[sim_mean_col]
            y_low = np.maximum(y_low, 0)
            y_high = np.maximum(y_high, 0)

            ax.errorbar(
                df_sim['Beta'],
                df_sim[sim_mean_col],
                yerr=[y_low, y_high],
                fmt='none',
                ecolor=color_sim,
                alpha=0.4,
                capsize=2,
                capthick=1,
                elinewidth=1
            )

        ax.set_title(f'{age_label}', fontsize=11, fontweight='bold')
        ax.set_xlabel(r'$\beta$', fontsize=9)
        ax.set_ylabel('Final epidemic size', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.legend(loc='best', fontsize=8, framealpha=0.8)

    plt.suptitle(f'{country_name}: Theory vs Simulation (16 Age Groups)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = f'{country_name}_16_groups_theory_vs_simulation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Generated: {output_file}")


def check_files_exist():
    """Check if all required files exist"""
    countries = ["Germany", "Uganda", "Qatar", "Monaco"]

    print("\nChecking files:")
    print("-" * 70)

    for country in countries:
        sim_file = f"Gillespie_SIR_{country}.csv"
        theory_file = f"Poison_beta_{country}.csv"

        sim_exists = os.path.exists(sim_file)
        theory_exists = os.path.exists(theory_file)

        status = "✓" if (sim_exists and theory_exists) else "✗"
        print(f"{status} {country}:")
        print(f"     Simulation: {sim_file} ({sim_exists})")
        print(f"     Theory: {theory_file} ({theory_exists})")

    print("-" * 70)


if __name__ == "__main__":
    # Check files first
    check_files_exist()

    # Plot 16 age groups comparison (all countries)
    plot_16_age_groups_comparison()

    # Optional: Plot single country detailed 16 age groups
    # plot_single_country_16_groups("Germany")