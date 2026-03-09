import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from scipy.stats import poisson, nbinom
import warnings

warnings.filterwarnings('ignore')

# ================= 路径设置 =================
GEXF_DIR = Path("gexf_networks")
OUTPUT_FILE = Path("degree_distribution_comparison.png")

# ================= 国家配置 =================
COUNTRIES = [
    {"name": "Uganda", "color": '#2E86AB'},
    {"name": "Qatar", "color": '#A23B72'},
    {"name": "Monaco", "color": '#F18F01'},
    {"name": "Germany", "color": '#C73E1D'},
]

# ================= 读取网络并计算度 =================
def load_degrees(country_name):

    possible_files = [
        GEXF_DIR / f"{country_name}.gexf",
        GEXF_DIR / f"{country_name.lower()}.gexf",
        GEXF_DIR / f"{country_name}_network.gexf",
        GEXF_DIR / f"network_{country_name}.gexf",
    ]

    gexf_file = None
    for f in possible_files:
        if f.exists():
            gexf_file = f
            break

    if gexf_file is None:
        print(f"No file found for {country_name}")
        return None

    print(f"Reading {gexf_file}")

    G = nx.read_gexf(str(gexf_file))

    if G.is_directed():
        G = G.to_undirected()

    degrees = [d for _, d in G.degree()]
    degrees = [d for d in degrees if d > 0]

    return np.array(degrees)


# ================= 拟合分布 =================
def fit_distributions(degrees):

    mu = np.mean(degrees)
    sigma2 = np.var(degrees)

    poisson_lambda = mu

    if sigma2 > mu:
        p = mu / sigma2
        r = mu * p / (1 - p)
    else:
        p = 0.5
        r = mu * 2

    return poisson_lambda, r, p, mu, np.sqrt(sigma2)


# ================= 生成图 =================
def create_degree_distribution_plot(country_data):

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (country, data) in enumerate(country_data.items()):

        ax = axes[i]
        degrees = data["degrees"]
        color = data["color"]

        lam, r, p, mean_k, std_k = data["fit"]

        max_degree = min(100, int(np.max(degrees)) + 1)

        hist, bins = np.histogram(
            degrees,
            bins=np.arange(0, max_degree + 1),
            density=True
        )

        centers = (bins[:-1] + bins[1:]) / 2

        ax.bar(
            centers,
            hist,
            width=0.8,
            alpha=0.7,
            color=color,
            edgecolor="black",
            label="Empirical Distribution"
        )

        k = np.arange(0, max_degree)

        poisson_prob = poisson.pmf(k, lam)
        ax.plot(k, poisson_prob, "r--", linewidth=2.5, label="Poisson")

        nb_prob = nbinom.pmf(k, r, p)
        ax.plot(k, nb_prob, "g-.", linewidth=2.5, label="Negative Binomial")

        ax.set_title(country, fontsize=16, fontweight="bold", color=color)
        ax.set_xlabel("Degree (k)")
        ax.set_ylabel("Probability P(k)")

        ax.set_xlim(-0.5, min(50, max_degree))
        ax.set_ylim(0, min(0.5, np.max(hist) * 1.5))

        text = f"N={len(degrees)}\n⟨k⟩={mean_k:.2f}\nσ={std_k:.2f}"
        ax.text(
            0.65, 0.95, text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.9)
        )

        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend()

    plt.suptitle(
        "Degree Distribution Comparison: Empirical vs Poisson & Negative Binomial",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to: {OUTPUT_FILE}")


# ================= 主程序 =================
def main():

    country_data = {}

    for c in COUNTRIES:

        degrees = load_degrees(c["name"])

        if degrees is None:
            continue

        fit = fit_distributions(degrees)

        country_data[c["name"]] = {
            "degrees": degrees,
            "fit": fit,
            "color": c["color"]
        }

    create_degree_distribution_plot(country_data)


if __name__ == "__main__":
    main()
