import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcdefaults()

FIGSIZE = (8,6)
DPI = 300

RESULTS_DIR = Path("../results_by_country")
REFERENCE_COUNTRIES = ["Uganda", "Qatar", "Monaco", "Germany"]
TARGET_FINAL_SIZE = 0.8
OUTPUT_DIR = Path(".")

age_labels = [
    '0-4', '5-9', '10-14', '15-19',
    '20-24', '25-29', '30-34', '35-39',
    '40-44', '45-49', '50-54', '55-59',
    '60-64', '65-69', '70-74', '75+'
]

COLORS = {
    'Uganda': '#2E86AB',
    'Qatar': '#A23B72',
    'Monaco': '#F18F01',
    'Germany': '#C73E1D',
    'Mean': '#2C3E50',
    'Std Range': '#3498DB',
    'MinMax Range': '#E74C3C'
}

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
})

ref_df = pd.read_csv(RESULTS_DIR / "Poison_beta_India.csv")
ref_df['diff'] = abs(ref_df['R_all'] - TARGET_FINAL_SIZE)
calibrated_beta = ref_df.loc[ref_df['diff'].idxmin(), 'Beta']

countries_data = {}
for c in REFERENCE_COUNTRIES:
    df = pd.read_csv(RESULTS_DIR / f"Poison_beta_{c}.csv")
    row = df.iloc[(df['Beta'] - calibrated_beta).abs().idxmin()]
    age_rates = [row[f'R{i + 1}'] for i in range(16)]

    countries_data[c] = {
        'R0': row['R_0'],
        'Final_Size': row['R_all'],
        'Age_Rates': age_rates
    }

all_csv_files = list(RESULTS_DIR.glob("Poison_beta_*.csv"))
all_age_rates = []
all_r0_values = []
all_final_sizes = []
all_country_names = []

for csv_file in all_csv_files:
    country_name = csv_file.stem.replace('Poison_beta_', '')
    all_country_names.append(country_name)

    try:
        df = pd.read_csv(csv_file)
        row = df.iloc[(df['Beta'] - calibrated_beta).abs().idxmin()]
        age_rates = [row[f'R{i + 1}'] for i in range(16)]

        all_age_rates.append(age_rates)
        all_r0_values.append(row['R_0'])
        all_final_sizes.append(row['R_all'])
    except Exception as e:
        print(f"Error processing {country_name}: {e}")
        continue

all_age_rates = np.array(all_age_rates)
mean_age_rates = np.mean(all_age_rates, axis=0)
std_age_rates = np.std(all_age_rates, axis=0)
min_age_rates = np.min(all_age_rates, axis=0)
max_age_rates = np.max(all_age_rates, axis=0)

mean_r0 = np.mean(all_r0_values)
std_r0 = np.std(all_r0_values)
min_r0 = np.min(all_r0_values)
max_r0 = np.max(all_r0_values)

mean_finalsize = np.mean(all_final_sizes)
std_finalsize = np.std(all_final_sizes)
min_finalsize = np.min(all_final_sizes)
max_finalsize = np.max(all_final_sizes)

print(f"Statistics for ALL countries (n={len(all_r0_values)}):")
print(f"R0 - Mean: {mean_r0:.3f}, Std: {std_r0:.3f}, Range: [{min_r0:.3f}, {max_r0:.3f}]")
print(f"Final Size - Mean: {mean_finalsize:.3f}, Std: {std_finalsize:.3f}, Range: [{min_finalsize:.3f}, {max_finalsize:.3f}]")

fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()

x = np.arange(len(REFERENCE_COUNTRIES) + 1)
width = 0.35

countries_list = REFERENCE_COUNTRIES + ['Mean (All\nCountries)']
r0_values = [countries_data[c]['R0'] for c in REFERENCE_COUNTRIES] + [mean_r0]
finalsize_values = [countries_data[c]['Final_Size'] for c in REFERENCE_COUNTRIES] + [mean_finalsize]

bars1 = ax1.bar(
    x - width / 2,
    r0_values,
    width,
    label=r"$R_0$",
    color=['#4C72B0'] * len(REFERENCE_COUNTRIES) + ['#4C72B0'],
    alpha=0.8,
    linewidth=1
)

bars2 = ax2.bar(
    x + width / 2,
    finalsize_values,
    width,
    label="Final epidemic size",
    color=['#DD8452'] * len(REFERENCE_COUNTRIES) + ['#DD8452'],
    alpha=0.8,
    linewidth=1
)

ax1.errorbar(x[-1] - width / 2, mean_r0, yerr=std_r0,
             fmt='none', ecolor='black', capsize=5, capthick=2, linewidth=2)
ax2.errorbar(x[-1] + width / 2, mean_finalsize, yerr=std_finalsize,
             fmt='none', ecolor='black', capsize=5, capthick=2, linewidth=2)

for i, (bar, val) in enumerate(zip(bars1, r0_values)):
    height = bar.get_height()
    offset = 0.08
    ax1.text(bar.get_x() + bar.get_width() / 2., height + offset,
             f'{val:.2f}', ha='center', va='center', fontsize=10)

for i, (bar, val) in enumerate(zip(bars2, finalsize_values)):
    height = bar.get_height()
    offset = 0.02
    ax2.text(bar.get_x() + bar.get_width() / 2., height + offset,
             f'{val:.3f}', ha='center', va='center', fontsize=10)

ax1.annotate(f'Range: [{min_r0:.2f}, {max_r0:.2f}]',
             xy=(x[-1] - width / 2, mean_r0 - std_r0+1.15 ), xytext=(0, 0),
             textcoords='offset points', fontsize=9, ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#4C72B0', alpha=0.15))

ax2.annotate(f'Range: [{min_finalsize:.3f}, {max_finalsize:.3f}]',
             xy=(x[-1] + width / 2, mean_finalsize - std_finalsize +0.15), xytext=(0, 0),
             textcoords='offset points', fontsize=9, ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#DD8452', alpha=0.15))

ax1.set_xlabel("Country", labelpad=5)
ax1.set_ylabel(r"Basic reproduction number $R_0$", color='#4C72B0')
ax2.set_ylabel("Final epidemic size", color='#DD8452')

ax1.tick_params(axis='y', labelcolor='#4C72B0')
ax2.tick_params(axis='y', labelcolor='#DD8452')

ax1.set_xticks(x)
ax1.set_xticklabels(countries_list, rotation=30, ha='right')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.9, fontsize=11)

ax1.grid(axis="y", alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "fig_r0_finalsize_dual_axis_with_stats.png",
    dpi=DPI,
    bbox_inches="tight"
)
plt.close()
print("Figure 1 saved: fig_r0_finalsize_dual_axis_with_stats.png")

fig, ax = plt.subplots(figsize=(9, 7))

x_age = np.arange(16)

marker_map = {
    'Uganda': 'o',
    'Qatar': 's',
    'Monaco': '^',
    'Germany': 'D',
}

for c in REFERENCE_COUNTRIES:
    ax.plot(
        x_age,
        countries_data[c]['Age_Rates'],
        marker=marker_map[c],
        linewidth=2.5,
        markersize=8,
        markerfacecolor=COLORS[c],
        markeredgecolor='black',
        markeredgewidth=0.8,
        label=c,
        color=COLORS[c],
        zorder=4
    )

ax.plot(
    x_age,
    mean_age_rates,
    marker='P',
    linewidth=3,
    markersize=9,
    markerfacecolor=COLORS['Mean'],
    markeredgecolor='black',
    markeredgewidth=0.8,
    label=f'Mean (All countries, n={len(all_r0_values)})',
    color=COLORS['Mean'],
    linestyle='-',
    zorder=5
)

ax.fill_between(
    x_age,
    mean_age_rates - std_age_rates,
    mean_age_rates + std_age_rates,
    alpha=0.3,
    color=COLORS['Std Range'],
    label=f'Mean +/- Standard Deviation',
    zorder=2
)

ax.fill_between(
    x_age,
    min_age_rates,
    max_age_rates,
    alpha=0.15,
    color=COLORS['MinMax Range'],
    label=f'Min-Max Range (n={len(all_r0_values)})',
    zorder=1
)

ax.set_xticks(x_age)
ax.set_xticklabels(age_labels, rotation=45, ha='right')

ax.set_xlabel("Age group")
ax.set_ylabel("Final epidemic size")

ax.legend(loc='best', framealpha=0.9, ncol=2)
ax.grid(alpha=0.3, linestyle='--', axis='both')

stats_text = f'All Countries Statistics (n={len(all_r0_values)})\n' + \
             f'Age groups - Mean range: [{min(mean_age_rates):.3f}, {max(mean_age_rates):.3f}]\n' + \
             f'Overall std range: [{min(std_age_rates):.3f}, {max(std_age_rates):.3f}]'
ax.text(0.98, 0.9, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8),
        fontfamily='monospace')

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "fig_age_rate_line_with_stats.png",
    dpi=DPI,
    bbox_inches="tight"
)
plt.close()
print("Figure 2 saved: fig_age_rate_line_with_stats.png")



print("\n" + "=" * 60)
print("All figures have been generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 60)
