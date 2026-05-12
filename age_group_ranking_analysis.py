import pandas as pd
from pathlib import Path
import numpy as np

# ================= CONFIGURATION =================
RESULTS_DIR = Path("results_by_country")
REFERENCE_COUNTRY = "India"
TARGET_FINAL_SIZE = 0.8
OUTPUT_DIR = RESULTS_DIR / "age_group_analysis222"

# ================= AGE GROUP DEFINITIONS =================
YOUNG_AGE_GROUPS = [0, 1, 2, 3]          # 0-19
MIDDLE_AGE_GROUPS = [4, 5, 6, 7]         # 20-39
MIDDLE_2_AGE_GROUPS = [8, 9, 10, 11]     # 40-59
OLD_AGE_GROUPS = [12, 13, 14, 15]        # 60+

age_group_labels = [
    "0-4 years", "5-9 years", "10-14 years", "15-19 years",
    "20-24 years", "25-29 years", "30-34 years", "35-39 years",
    "40-44 years", "45-49 years", "50-54 years", "55-59 years",
    "60-64 years", "65-69 years", "70-74 years", "75+ years"
]

young_labels = [age_group_labels[i] for i in YOUNG_AGE_GROUPS]
middle_labels = [age_group_labels[i] for i in MIDDLE_AGE_GROUPS]
middle2_labels = [age_group_labels[i] for i in MIDDLE_2_AGE_GROUPS]
old_labels = [age_group_labels[i] for i in OLD_AGE_GROUPS]

print("Age group definitions:")
print(f"  Young (0-19): {', '.join(young_labels)}")
print(f"  Middle (20-39): {', '.join(middle_labels)}")
print(f"  Middle-old (40-59): {', '.join(middle2_labels)}")
print(f"  Old (60+): {', '.join(old_labels)}")

# ================= CREATE OUTPUT DIRECTORY =================
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "age_group_rankings").mkdir(exist_ok=True)

# ================= LOAD DATA =================
result_files = list(RESULTS_DIR.glob("Poison_beta_*.csv"))
ref_df = pd.read_csv(RESULTS_DIR / f"Poison_beta_{REFERENCE_COUNTRY}.csv")
ref_df['diff'] = abs(ref_df['R_all'] - TARGET_FINAL_SIZE)
calibrated_beta = ref_df.loc[ref_df['diff'].idxmin(), 'Beta']

all_country_data = []

for f in result_files:
    df = pd.read_csv(f)
    df['beta_diff'] = abs(df['Beta'] - calibrated_beta)
    row = df.loc[df['beta_diff'].idxmin()]

    data = {
        'Country': f.stem.replace("Poison_beta_", ""),
        'Beta_Used': row['Beta'],
        'R0': row['R_0'],
        'Final_Size': row['R_all']
    }

    for i in range(16):
        data[f'Age_{i+1}'] = row[f'R{i+1}']

    data['Young_Avg'] = np.mean([row[f'R{i+1}'] for i in YOUNG_AGE_GROUPS])
    data['Middle_Avg'] = np.mean([row[f'R{i+1}'] for i in MIDDLE_AGE_GROUPS])
    data['Middle2_Avg'] = np.mean([row[f'R{i+1}'] for i in MIDDLE_2_AGE_GROUPS])
    data['Old_Avg'] = np.mean([row[f'R{i+1}'] for i in OLD_AGE_GROUPS])

    all_country_data.append(data)

main_df = pd.DataFrame(all_country_data)

# ================= RANKING FILES FOR EACH AGE CATEGORY =================

# Young
sorted_young = main_df.sort_values('Young_Avg', ascending=False).reset_index(drop=True)
sorted_young['Rank'] = range(1, len(sorted_young) + 1)
sorted_young.to_csv(
    OUTPUT_DIR / "Young_Age_Category_Ranking.csv",
    index=False
)

# Middle
sorted_middle = main_df.sort_values('Middle_Avg', ascending=False).reset_index(drop=True)
sorted_middle['Rank'] = range(1, len(sorted_middle) + 1)
sorted_middle.to_csv(
    OUTPUT_DIR / "Middle_Age_Category_Ranking.csv",
    index=False
)

# Middle-old
sorted_middle2 = main_df.sort_values('Middle2_Avg', ascending=False).reset_index(drop=True)
sorted_middle2['Rank'] = range(1, len(sorted_middle2) + 1)
sorted_middle2.to_csv(
    OUTPUT_DIR / "Middle2_Age_Category_Ranking.csv",
    index=False
)

# Old
sorted_old = main_df.sort_values('Old_Avg', ascending=False).reset_index(drop=True)
sorted_old['Rank'] = range(1, len(sorted_old) + 1)
sorted_old.to_csv(
    OUTPUT_DIR / "Old_Age_Category_Ranking.csv",
    index=False
)

# ================= OUTPUT STRUCTURE DESCRIPTION =================
print(f"""
Output folder structure:
{OUTPUT_DIR}/
├── Young_Age_Category_Ranking.csv
├── Middle_Age_Category_Ranking.csv
├── Middle2_Age_Category_Ranking.csv
└── Old_Age_Category_Ranking.csv

Age group definitions:
  Young (0-19): {', '.join(young_labels)}
  Middle (20-39): {', '.join(middle_labels)}
  Middle-old (40-59): {', '.join(middle2_labels)}
  Old (60+): {', '.join(old_labels)}

All age group ranking files have been generated with consistent format.
""")