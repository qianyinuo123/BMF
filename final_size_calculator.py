import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import math
import warnings

warnings.filterwarnings('ignore')

# ================= GLOBAL PARAMETERS =================
times = 1000
gamma = 0.333333333333333333

# ================= PATH SETUP =================
rho_file = Path("summary-contact-matrix-R0.xlsx")
c_dir = Path("processed_matrices")
N_dir = Path("population_fraction_processed_csv")
output_dir = Path("results_by_country")
output_dir.mkdir(exist_ok=True)

# ================= READ rho_C =================
rho_df = pd.read_excel(rho_file, header=None)
rho_df.columns = ["Country", "rho_C"]
rho_dict = dict(zip(rho_df["Country"], rho_df["rho_C"]))

# ================= PROCESS ONLY FOUR TARGET COUNTRIES =================
target_countries = ["Uganda", "Qatar", "Monaco", "Germany"]

print("=" * 60)
print("Processing only four target countries:")
for country in target_countries:
    if country in rho_dict:
        print(f"  {country}: rho(C) = {rho_dict[country]:.4f}")
    else:
        print(f"  {country}: not found in rho file, using default value")
        if country == "Uganda":
            rho_dict[country] = 0.9752
        elif country == "Qatar":
            rho_dict[country] = 0.9506
        elif country == "Monaco":
            rho_dict[country] = 0.7228
        elif country == "Germany":
            rho_dict[country] = np.mean([0.9752, 0.9506, 0.7228])
        print(f"    using: rho(C) = {rho_dict[country]:.4f}")
print("=" * 60)

# ================= ITERATE OVER FOUR TARGET COUNTRIES =================
for country in target_countries:

    if country not in rho_dict:
        print(f"Skip {country}: rho(C) value not found")
        continue

    rho_C = rho_dict[country]

    c_file = c_dir / f"{country}.pkl"
    N_file = N_dir / f"{country}.csv"

    file_errors = []
    if not c_file.exists():
        file_errors.append(f"contact matrix file: {c_file}")
    if not N_file.exists():
        file_errors.append(f"population fraction file: {N_file}")

    if file_errors:
        print(f"Skip {country}: missing files")
        for error in file_errors:
            print(f"  - {error}")
        continue

    print(f"\n{'=' * 60}")
    print(f"Processing: {country}")
    print(f"rho(C) = {rho_C:.4f}")
    print(f"{'=' * 60}")

    # ---------- LOAD CONTACT MATRIX ----------
    try:
        with open(c_file, "rb") as f:
            c = pickle.load(f)

        if c.shape != (16, 16):
            print(f"Warning: {country} contact matrix shape is {c.shape}, adjusting to 16x16")
            if c.shape[0] >= 16 and c.shape[1] >= 16:
                c = c[:16, :16]
            else:
                print(f"  matrix too small, creating 16x16 zero matrix")
                c = np.zeros((16, 16))

        zero_elements = np.sum(np.abs(c) < 1e-12)
        total_elements = 16 * 16
        print(
            f"Contact matrix: shape={c.shape}, zero elements={zero_elements}/{total_elements} ({zero_elements / total_elements:.1%})")

    except Exception as e:
        print(f"Failed to load contact matrix: {e}")
        continue

    # ---------- LOAD POPULATION FRACTION ----------
    try:
        N_df = pd.read_csv(N_file)
        if len(N_df.columns) >= 16:
            N = N_df.iloc[0, :16].to_numpy(dtype=float)
        else:
            N = N_df.iloc[0].to_numpy(dtype=float)

        if len(N) < 16:
            N_full = np.zeros(16)
            N_full[:len(N)] = N
            N = N_full
        elif len(N) > 16:
            N = N[:16]

        N_sum = np.sum(N)
        if N_sum > 0:
            N = N / N_sum
        else:
            print(f"Warning: {country} population fraction sum is zero, using uniform distribution")
            N = np.ones(16) / 16

        print(f"Population fraction: sum = {np.sum(N):.6f}")

    except Exception as e:
        print(f"Failed to load population fraction: {e}")
        continue

    results = []

    # ================= SCAN BETA =================
    print(f"Scanning Beta values (0 to 0.4, step 0.001)...")

    for b in range(401):
        Beta = b / 1000.0
        R0 = Beta / (Beta + gamma) * rho_C

        theta = [[0.5] * 16 for _ in range(16)]
        phi = [[0.0] * 16 for _ in range(16)]
        phi1 = [[0.0] * 16 for _ in range(16)]

        converged = True
        for iteration in range(times):

            for j in range(16):
                for l in range(16):
                    exponent = -c[j][l] * (1.0 - theta[j][l])
                    phi[j][l] = math.exp(exponent)
                    phi1[j][l] = c[j][l] * phi[j][l]

            prod = [1.0] * 16
            for j in range(16):
                for l in range(16):
                    prod[j] *= phi[j][l]

            theta_old = [[theta[j][l] for l in range(16)] for j in range(16)]

            for j in range(16):
                for l in range(16):
                    if abs(c[l][j]) < 1e-12:
                        term1 = phi[l][j] if abs(phi[l][j]) > 1e-12 else 1.0
                    else:
                        term1 = phi1[l][j] / c[l][j]

                    if abs(phi[l][j]) < 1e-12:
                        term2 = 0.0
                    else:
                        term2 = prod[l] / phi[l][j]

                    theta[j][l] = (
                            gamma / (Beta + gamma)
                            + Beta / (Beta + gamma) * term1 * term2
                    )

            max_diff = 0.0
            for j in range(16):
                for l in range(16):
                    diff = abs(theta[j][l] - theta_old[j][l])
                    if diff > max_diff:
                        max_diff = diff

            if iteration > 10 and max_diff < 1e-8:
                print(f"  Beta={Beta:.3f}: converged after {iteration + 1} iterations")
                break

        S = [1.0] * 16
        R = [0.0] * 16

        for j in range(16):
            for l in range(16):
                exponent = -c[j][l] * (1.0 - theta[j][l])
                S[j] *= math.exp(exponent)
            R[j] = 1.0 - S[j]

        R_all = 0.0
        for j in range(16):
            R_all += R[j] * N[j]

        results.append([Beta, R0] + R + [R_all])

        if b % 50 == 0:
            print(f"  Progress: Beta={Beta:.3f}, R0={R0:.3f}, R_all={R_all:.4f}")

    columns = (
            ["Beta", "R_0"]
            + [f"R{i + 1}" for i in range(16)]
            + ["R_all"]
    )

    df_out = pd.DataFrame(results, columns=columns)

    out_file = output_dir / f"Poison_beta_{country}.csv"
    df_out.to_csv(out_file, index=False)

    out_excel = output_dir / f"Poison_beta_{country}.xlsx"
    df_out.to_excel(out_excel, index=False)

    print(f"\n{country} key results:")
    print("-" * 40)

    key_betas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    for beta_val in key_betas:
        beta_idx = int(beta_val * 1000)
        if beta_idx < len(results):
            row = results[beta_idx]
            print(f"  Beta={row[0]:.3f}: R0={row[1]:.3f}, R_all={row[-1]:.4f}")

    print(f"\nResults saved:")
    print(f"  CSV: {out_file}")
    print(f"  Excel: {out_excel}")

print("\n" + "=" * 60)
print("All target countries processed!")
print("=" * 60)

# ================= GENERATE SUMMARY REPORT =================
print("\nGenerating summary report...")

summary_data = []
for country in target_countries:
    if country not in rho_dict:
        continue

    result_file = output_dir / f"Poison_beta_{country}.csv"
    if result_file.exists():
        try:
            df_country = pd.read_csv(result_file)

            r0_1_row = df_country.iloc[(df_country['R_0'] - 1.0).abs().argsort()[:1]]
            r0_2_row = df_country.iloc[(df_country['R_0'] - 2.0).abs().argsort()[:1]]

            if not r0_1_row.empty:
                r0_1_beta = r0_1_row.iloc[0]['Beta']
                r0_1_R_all = r0_1_row.iloc[0]['R_all']
            else:
                r0_1_beta = np.nan
                r0_1_R_all = np.nan

            if not r0_2_row.empty:
                r0_2_beta = r0_2_row.iloc[0]['Beta']
                r0_2_R_all = r0_2_row.iloc[0]['R_all']
            else:
                r0_2_beta = np.nan
                r0_2_R_all = np.nan

            max_R_all = df_country['R_all'].max()
            max_R_all_beta = df_country.loc[df_country['R_all'].idxmax(), 'Beta']

            summary_data.append({
                'Country': country,
                'rho_C': rho_dict[country],
                'Beta_R0=1': r0_1_beta,
                'R_all_R0=1': r0_1_R_all,
                'Beta_R0=2': r0_2_beta,
                'R_all_R0=2': r0_2_R_all,
                'Max_R_all': max_R_all,
                'Beta_Max_R_all': max_R_all_beta
            })

        except Exception as e:
            print(f"Failed to read {country} results: {e}")

if summary_data:
    df_summary = pd.DataFrame(summary_data)
    summary_file = output_dir / "summary_four_countries.xlsx"
    df_summary.to_excel(summary_file, index=False)

    print(f"\nSummary report saved: {summary_file}")
    print("\nSummary results:")
    print(df_summary.to_string())
else:
    print("No summary report generated")