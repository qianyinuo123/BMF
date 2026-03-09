import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcdefaults()

FIGSIZE = (8, 6)
DPI = 300

RESULTS_DIR = Path("results_by_country")
REFERENCE_COUNTRIES = ["Uganda", "Qatar", "Monaco", "Germany"]
TARGET_FINAL_SIZE = 0.8
OUTPUT_DIR = RESULTS_DIR / "special_countries_visualization_split"
OUTPUT_DIR.mkdir(exist_ok=True)

age_labels = [
    '0-4','5-9','10-14','15-19',
    '20-24','25-29','30-34','35-39',
    '40-44','45-49','50-54','55-59',
    '60-64','65-69','70-74','75+'
]

COLORS = {
    'Uganda': '#2E86AB',
    'Qatar': '#A23B72',
    'Monaco': '#F18F01',
    'Germany': '#C73E1D',
}

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

ref_df = pd.read_csv(RESULTS_DIR / "Poison_beta_India.csv")
ref_df['diff'] = abs(ref_df['R_all'] - TARGET_FINAL_SIZE)
calibrated_beta = ref_df.loc[ref_df['diff'].idxmin(), 'Beta']

countries_data = {}
for c in REFERENCE_COUNTRIES:
    df = pd.read_csv(RESULTS_DIR / f"Poison_beta_{c}.csv")
    row = df.iloc[(df['Beta'] - calibrated_beta).abs().idxmin()]
    age_rates = [row[f'R{i+1}'] for i in range(16)]

    countries_data[c] = {
        'R0': row['R_0'],
        'Final_Size': row['R_all'],
        'Age_Rates': age_rates
    }

fig, ax1 = plt.subplots(figsize=FIGSIZE)
ax2 = ax1.twinx()

x = np.arange(len(REFERENCE_COUNTRIES))
width = 0.35

ax1.bar(
    x - width / 2,
    [countries_data[c]['R0'] for c in REFERENCE_COUNTRIES],
    width,
    label=r"$R_0$",
    color="#4C72B0"
)

ax2.bar(
    x + width / 2,
    [countries_data[c]['Final_Size'] for c in REFERENCE_COUNTRIES],
    width,
    label="Final epidemic size",
    color="#DD8452"
)

ax1.set_xlabel("Country")
ax1.set_ylabel(r"Basic reproduction number $R_0$")
ax2.set_ylabel("Final epidemic size")

ax1.set_xticks(x)
ax1.set_xticklabels(REFERENCE_COUNTRIES, rotation=30)

ax1.set_title("Transmission potential and epidemic outcome")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")

ax1.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "fig_r0_finalsize_dual_axis.png",
    dpi=DPI,
    bbox_inches="tight"
)
plt.close()

fig, ax = plt.subplots(figsize=FIGSIZE)

x_age = np.arange(16)

for c in REFERENCE_COUNTRIES:
    ax.plot(
        x_age,
        countries_data[c]['Age_Rates'],
        marker='o',
        linewidth=2.5,
        label=c,
        color=COLORS[c]
    )

ax.set_xticks(x_age)
ax.set_xticklabels(age_labels, rotation=45)

ax.set_xlabel("Age group")
ax.set_ylabel("Final epidemic size")
ax.set_title("Age Group Final Epidemic Size Distribution")

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "fig_age_rate_line.png",
    dpi=DPI,
    bbox_inches="tight"
)
plt.close()
