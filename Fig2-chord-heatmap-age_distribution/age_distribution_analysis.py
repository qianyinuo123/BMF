import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'

gexf_folder = ""
output_folder = "compact_age_analysis-new"

os.makedirs(output_folder, exist_ok=True)

SELECTED_COUNTRIES = ["Germany", "Uganda", "Qatar", "Monaco"]


def get_age_distribution(G, country):
    node_groups = nx.get_node_attributes(G, 'group')

    group_sizes = {}
    for group in node_groups.values():
        group_sizes[group] = group_sizes.get(group, 0) + 1

    return group_sizes


def plot_individual_age_distribution(country, group_sizes, output_folder):
    print(f"   Plotting {country}...")

    fig, ax = plt.subplots(figsize=(8, 6))

    uniform_color = '#2E86AB'

    ages = list(range(16))
    sizes = [group_sizes.get(age, 0) for age in ages]
    total = sum(sizes)
    percentages = [size / total * 100 for size in sizes] if total > 0 else [0] * 16

    bars = ax.bar(ages, percentages,
                  color=uniform_color,
                  alpha=0.8,
                  edgecolor='black',
                  linewidth=1.2,
                  width=0.7)

    ax.set_xlabel('Age Group', fontsize=22, fontweight='normal')
    ax.set_ylabel('Percentage (%)', fontsize=22, fontweight='normal')
    ax.set_xticks(ages)
    ax.set_xticklabels([f'{age}' for age in ages], fontsize=20)
    ax.tick_params(axis='y', labelsize=20)

    max_percent = max(percentages) if percentages else 0
    ax.set_ylim(0, min(100, max_percent * 1.2))

    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    ax.text(0.02, 0.98, f'Total: {total:,}',
            transform=ax.transAxes, fontsize=18, fontweight='normal',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    if sizes:
        max_age = np.argmax(sizes)
        max_percent_val = percentages[max_age]
        ax.text(0.98, 0.98, f'Max: Age {max_age} ({max_percent_val:.1f}%)',
                transform=ax.transAxes, fontsize=18, fontweight='normal',
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

    output_path = os.path.join(output_folder, f'{country}_age_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_statistical_summary(all_distributions, output_folder):
    print("Creating statistical summary...")

    summary_data = []

    for country, group_sizes in all_distributions.items():
        sizes = [group_sizes.get(age, 0) for age in range(16)]
        total = sum(sizes)

        if total > 0:
            percentages = [size / total * 100 for size in sizes]

            max_age = np.argmax(sizes)
            max_percent = percentages[max_age]

            young_pop = sum(sizes[0:6]) / total * 100
            middle_pop = sum(sizes[6:11]) / total * 100
            old_pop = sum(sizes[11:16]) / total * 100

            weighted_age = sum(age * sizes[age] for age in range(16)) / total

            summary_data.append({
                'Country': country,
                'Total Population': f'{total:,}',
                'Largest Age Group': f'Age {max_age}',
                'Largest %': f'{max_percent:.1f}%',
                'Young (0-5)': f'{young_pop:.1f}%',
                'Middle (6-10)': f'{middle_pop:.1f}%',
                'Old (11-15)': f'{old_pop:.1f}%',
                'Avg Age': f'{weighted_age:.1f}'
            })

    print("\n" + "=" * 90)
    print("AGE DISTRIBUTION STATISTICAL SUMMARY")
    print("=" * 90)

    if summary_data:
        print(f"{'Country':<12} {'Total':<14} {'Largest':<14} {'Largest%':<12} "
              f"{'Young%':<12} {'Middle%':<12} {'Old%':<12} {'Avg Age':<10}")
        print("-" * 100)

        for stats in summary_data:
            print(f"{stats['Country']:<12} {stats['Total Population']:<14} "
                  f"{stats['Largest Age Group']:<14} {stats['Largest %']:<12} "
                  f"{stats['Young (0-5)']:<12} {stats['Middle (6-10)']:<12} "
                  f"{stats['Old (11-15)']:<12} {stats['Avg Age']:<10}")

    print("=" * 90)

    if summary_data:
        import pandas as pd
        df_stats = pd.DataFrame(summary_data)
        stats_path = os.path.join(output_folder, 'age_statistics_summary.csv')
        df_stats.to_csv(stats_path, index=False)
        print(f"Saved statistics summary: {stats_path}")


def main():
    print("=" * 60)
    print("AGE GROUP DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print("Countries to analyze:")
    for i, country in enumerate(SELECTED_COUNTRIES, 1):
        print(f"  {i}. {country}")
    print("=" * 60)

    all_distributions = {}

    for country in SELECTED_COUNTRIES:
        file_path = os.path.join(gexf_folder, f"{country}.gexf")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            G = nx.read_gexf(file_path)
            group_sizes = get_age_distribution(G, country)

            if group_sizes:
                all_distributions[country] = group_sizes
                print(f"{country}: {sum(group_sizes.values()):,} total population")

        except Exception as e:
            print(f"Error processing {country}: {e}")

    if all_distributions:
        print(f"\nGenerating individual charts for {len(all_distributions)} countries...")

        for country, group_sizes in all_distributions.items():
            plot_individual_age_distribution(country, group_sizes, output_folder)

        create_statistical_summary(all_distributions, output_folder)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED!")
    print(f"Results saved in: {output_folder}")
    print("=" * 60)

    print("\nGenerated files:")
    print("-" * 40)
    for country in SELECTED_COUNTRIES:
        print(f"{country}_age_distribution.png")
    print("age_statistics_summary.csv")
    print("-" * 40)


if __name__ == "__main__":
    main()
