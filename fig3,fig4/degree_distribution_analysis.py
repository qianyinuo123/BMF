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
OUTPUT_DIR = RESULTS_DIR / "degree_distribution_analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

COUNTRIES = [
    {"name": "Uganda", "region": "Africa"},
    {"name": "Qatar", "region": "Asia"},
    {"name": "Monaco", "region": "Europe"},
    {"name": "Germany", "region": "Europe"},
]

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 14,
    'figure.titlesize': 18
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

        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'mean_degree': np.mean(degrees),
            'std_degree': np.std(degrees),
            'max_degree': np.max(degrees),
            'degrees': degrees,
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
                     f"D = {fits['dispersion']:.2f}\n"
                     )

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


def main():
    print("=" * 60)
    print("DEGREE DISTRIBUTION ANALYSIS (Figure 5)")
    print("=" * 60)

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

    print("\n" + "-" * 60)
    create_degree_distribution_chart(country_stats)

    print("\n" + "-" * 60)
    print("DETAILED RESULTS")
    print("-" * 60)

    for country_name, data in country_stats.items():
        fits = data['fits']
        ks = data['ks_test']
        print(f"\n{country_name}:")
        print(f"  • Network: {data['stats']['num_nodes']:,} nodes, {data['stats']['num_edges']:,} edges")
        print(f"  • Mean degree ⟨k⟩ = {fits['mean']:.2f}, σ = {data['stats']['std_degree']:.2f}")
        print(f"  • Dispersion D = σ²/μ = {fits['dispersion']:.2f}")
        print(f"  • Best fit: {ks['better_fit']} (KS: Poisson={ks['ks_poisson']:.3f}, NB={ks['ks_nb']:.3f})")

    print("\n" + "=" * 60)
    print(f"✅ Figure 5 saved to: {OUTPUT_DIR / 'degree_distribution_comparison.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
