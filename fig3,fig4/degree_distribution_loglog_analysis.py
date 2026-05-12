import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import poisson, nbinom, ks_2samp
import warnings

warnings.filterwarnings('ignore')

GEXF_DIR = Path("..\gexf_networks")
RESULTS_DIR = Path("results_by_country")
OUTPUT_DIR = RESULTS_DIR / "degree_distribution_analysis-new-loglog"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

COUNTRIES = [
    {"name": "Uganda", "region": "Africa"},
    {"name": "Qatar", "region": "Asia"},
    {"name": "Monaco", "region": "Europe"},
    {"name": "Germany", "region": "Europe"},
]

COUNTRY_COLORS = {
    "Uganda": "#2E86AB",
    "Qatar": "#A23B72",
    "Monaco": "#F18F01",
    "Germany": "#C73E1D",
}

COUNTRY_MARKERS = {
    "Uganda": 'o',
    "Qatar": 's',
    "Monaco": '^',
    "Germany": 'D',
}

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 20
})


def analyze_gexf_file(country_name):
    possible_files = [
        GEXF_DIR / f"{country_name}.gexf",
        GEXF_DIR / f"{country_name.lower()}.gexf",
        GEXF_DIR / f"{country_name}_network.gexf",
    ]

    gexf_file = None
    for file in possible_files:
        if file.exists():
            gexf_file = file
            break

    if gexf_file is None:
        print(f"  Warning: No GEXF file found for {country_name}")
        return None

    try:
        G = nx.read_gexf(str(gexf_file))

        if G.is_directed():
            G = G.to_undirected()

        degrees = [d for n, d in G.degree()]
        degrees = [d for d in degrees if d > 0]

        if len(degrees) == 0:
            return None

        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1

        unique_degrees = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in unique_degrees]
        probabilities = [c / len(degrees) for c in counts]

        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'mean_degree': np.mean(degrees),
            'std_degree': np.std(degrees),
            'max_degree': np.max(degrees),
            'min_degree': np.min(degrees),
            'degrees': degrees,
            'unique_degrees': unique_degrees,
            'probabilities': probabilities,
        }
        return stats

    except Exception as e:
        print(f"  Error analyzing {country_name}: {e}")
        return None


def fit_distributions(degrees):
    degrees = np.array(degrees)
    mu = np.mean(degrees)
    sigma2 = np.var(degrees)

    poisson_lambda = mu

    if sigma2 > mu and mu > 0:
        p = mu / sigma2
        r = mu * p / (1 - p)
    else:
        p = 0.5
        r = mu * 2 if mu > 0 else 1

    return {
        'poisson': {'lambda': float(poisson_lambda)},
        'negative_binomial': {'r': float(r), 'p': float(p)},
        'mean': float(mu),
        'variance': float(sigma2),
        'dispersion': float(sigma2 / mu) if mu > 0 else 1.0
    }


def calculate_ks_test(degrees, fit_params):
    try:
        n_samples = min(10000, len(degrees) * 10)
        poisson_sample = np.random.poisson(fit_params['poisson']['lambda'], n_samples)
        nb_sample = np.random.negative_binomial(
            fit_params['negative_binomial']['r'],
            fit_params['negative_binomial']['p'],
            n_samples
        )
        ks_poisson = ks_2samp(degrees, poisson_sample).statistic
        ks_nb = ks_2samp(degrees, nb_sample).statistic
        return {
            'ks_poisson': float(ks_poisson),
            'ks_nb': float(ks_nb),
            'better_fit': 'Negative Binomial' if ks_nb < ks_poisson else 'Poisson'
        }
    except:
        return {'ks_poisson': 1.0, 'ks_nb': 1.0, 'better_fit': 'Unknown'}


def create_degree_distribution_chart(country_stats):
    print("\nCreating degree distribution chart...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    uniform_color = '#2E86AB'

    for idx, (country_name, data) in enumerate(country_stats.items()):
        if idx >= 4:
            break

        ax = axes[idx]
        degrees = data['stats']['degrees']
        fits = data['fits']

        max_degree = int(np.max(degrees)) + 1
        hist, bin_edges = np.histogram(degrees, bins=np.arange(0, max_degree + 1), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ax.bar(bin_centers, hist, width=0.8, alpha=0.7, color=uniform_color,
               label='Empirical', edgecolor='black', linewidth=0.5)

        k_values = np.arange(0, max_degree)

        poisson_probs = poisson.pmf(k_values, fits['poisson']['lambda'])
        ax.plot(k_values, poisson_probs, 'r--', linewidth=2.5, label='Poisson', alpha=0.8)

        nb_probs = nbinom.pmf(k_values, fits['negative_binomial']['r'],
                              fits['negative_binomial']['p'])
        ax.plot(k_values, nb_probs, 'g-.', linewidth=2.5, label='Negative Binomial', alpha=0.8)

        max_hist = np.max(hist) if len(hist) > 0 else 0.1
        max_poisson = np.max(poisson_probs) if len(poisson_probs) > 0 else 0
        max_nb = np.max(nb_probs) if len(nb_probs) > 0 else 0
        y_max = max(max_hist, max_poisson, max_nb) * 1.3
        if y_max < 0.1:
            y_max = 0.15

        ax.set_xlim(-0.5, min(50, max_degree * 1.1))
        ax.set_ylim(0, y_max)

        ax.set_title(f"{country_name}", fontsize=20, pad=10)
        ax.set_xlabel('Degree (k)', fontsize=16)
        ax.set_ylabel('Probability P(k)', fontsize=16)

        info_text = (f"N = {data['stats']['num_nodes']:,}\n"
                     f"⟨k⟩ = {data['stats']['mean_degree']:.2f}\n"
                     f"σ = {data['stats']['std_degree']:.2f}\n"
                     f"D = {fits['dispersion']:.2f}")

        ax.text(0.68, 0.95, info_text, transform=ax.transAxes, fontsize=15,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.9, edgecolor='gray'))

        ax.grid(True, alpha=0.2, linestyle='--')

        if idx == 0:
            ax.legend(loc='upper left', fontsize=15, framealpha=0.9)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "degree_distribution_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Degree distribution chart saved to: {output_file}")
    return output_file


def create_loglog_plot(country_stats):
    print("\nCreating log-log degree distribution plot...")

    fig, ax = plt.subplots(figsize=(6, 5))

    for country_name, data in country_stats.items():
        unique_degrees = np.array(data['stats']['unique_degrees'])
        probabilities = np.array(data['stats']['probabilities'])
        color = COUNTRY_COLORS.get(country_name, '#2E86AB')
        marker = COUNTRY_MARKERS.get(country_name, 'o')

        ax.loglog(unique_degrees, probabilities, marker,
                  markersize=7, color=color, alpha=0.8,
                  label=f"{country_name}",
                  markeredgecolor='none',
                  linestyle='None')

        valid_mask = (unique_degrees >= 1) & (probabilities > 0)
        valid_degrees = unique_degrees[valid_mask]
        valid_probs = probabilities[valid_mask]

        if len(valid_degrees) > 5:
            log_degrees = np.log(valid_degrees)
            log_probs = np.log(valid_probs)

            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_degrees, log_probs
            )

            x_fit = valid_degrees
            y_fit = np.exp(intercept + slope * np.log(x_fit))
            ax.loglog(x_fit, y_fit, '-', linewidth=1.2, alpha=0.5, color=color, linestyle='--')

    ax.set_xlabel('Degree (k)', fontsize=16, fontweight='normal')
    ax.set_ylabel('Probability P(k)', fontsize=16, fontweight='normal')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6, which='both')

    ax.legend(loc='lower left', fontsize=14, framealpha=0.9, edgecolor='black')

    ax.set_xlim(0.8, None)
    ax.set_ylim(1e-6, None)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "degree_distribution_loglog.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Log-log plot saved to: {output_file}")
    return output_file


def create_statistics_table(country_stats):
    print("\nCreating statistics table...")

    summary_data = []
    for country_name, data in country_stats.items():
        stats = data['stats']
        fits = data['fits']
        ks = data['ks_test']

        summary_data.append({
            'Country': country_name,
            'Nodes': stats['num_nodes'],
            'Edges': stats['num_edges'],
            'Mean_Degree': f"{stats['mean_degree']:.2f}",
            'Std_Degree': f"{stats['std_degree']:.2f}",
            'Dispersion': f"{fits['dispersion']:.3f}",
            'Max_Degree': stats['max_degree'],
            'Min_Degree': stats['min_degree'],
            'Poisson_Lambda': f"{fits['poisson']['lambda']:.2f}",
            'NB_r': f"{fits['negative_binomial']['r']:.2f}",
            'NB_p': f"{fits['negative_binomial']['p']:.3f}",
            'KS_Poisson': f"{ks['ks_poisson']:.4f}",
            'KS_NB': f"{ks['ks_nb']:.4f}",
            'Best_Fit': ks['better_fit']
        })

    df_summary = pd.DataFrame(summary_data)
    csv_file = OUTPUT_DIR / "degree_distribution_statistics.csv"
    df_summary.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"  ✓ Statistics table saved to: {csv_file}")

    return df_summary


def save_detailed_degree_data(country_stats):
    print("\nSaving detailed degree distribution data...")

    for country_name, data in country_stats.items():
        unique_degrees = data['stats']['unique_degrees']
        probabilities = data['stats']['probabilities']

        df_degree = pd.DataFrame({
            'degree': unique_degrees,
            'probability': probabilities,
            'cumulative_probability': np.cumsum(probabilities)
        })

        csv_file = OUTPUT_DIR / f"{country_name}_degree_distribution.csv"
        df_degree.to_csv(csv_file, index=False)
        print(f"  ✓ {country_name} degree data saved to: {csv_file}")


def main():
    print("=" * 70)
    print("DEGREE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    if not GEXF_DIR.exists():
        print(f"ERROR: GEXF directory not found: {GEXF_DIR}")
        return

    print("\nAnalyzing GEXF files...")
    country_stats = {}

    for country in COUNTRIES:
        print(f"\nAnalyzing {country['name']}...")
        stats = analyze_gexf_file(country['name'])
        if stats is not None:
            fits = fit_distributions(stats['degrees'])
            ks_test = calculate_ks_test(stats['degrees'], fits)
            country_stats[country['name']] = {
                'stats': stats,
                'fits': fits,
                'ks_test': ks_test
            }
            print(f"  ✓ Mean degree: {stats['mean_degree']:.2f}, D: {fits['dispersion']:.2f}")

    if len(country_stats) == 0:
        print("\nNo valid GEXF files could be analyzed!")
        return

    print("\n" + "-" * 70)

    create_degree_distribution_chart(country_stats)

    create_loglog_plot(country_stats)

    create_statistics_table(country_stats)

    save_detailed_degree_data(country_stats)

    print("\n" + "-" * 70)
    print("DETAILED RESULTS")
    print("-" * 70)

    for country_name, data in country_stats.items():
        fits = data['fits']
        ks = data['ks_test']
        print(f"\n{country_name}:")
        print(f"  • Network: {data['stats']['num_nodes']:,} nodes, {data['stats']['num_edges']:,} edges")
        print(f"  • Mean degree ⟨k⟩ = {fits['mean']:.2f}, σ = {data['stats']['std_degree']:.2f}")
        print(f"  • Dispersion D = σ²/μ = {fits['dispersion']:.3f}")
        print(f"  • Best fit: {ks['better_fit']} (KS: Poisson={ks['ks_poisson']:.4f}, NB={ks['ks_nb']:.4f})")

    print("\n" + "=" * 70)
    print("Analysis completed successfully!")
    print(f"utput directory: {OUTPUT_DIR}")
    print("=" * 70)

    print("\nGenerated files:")
    print("-" * 50)
    print("  • degree_distribution_comparison.png")
    print("  • degree_distribution_loglog.png")
    print("  • degree_distribution_statistics.csv")
    print("  • [Country]_degree_distribution.csv")
    print("-" * 50)


if __name__ == "__main__":
    main()
