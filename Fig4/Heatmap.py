# intra_group_heatmap_final.py
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_generate_intra_group_heatmap(country_name, gexf_folder="gexf_networks",
                                          output_folder="intra_group_analysis"):

    os.makedirs(output_folder, exist_ok=True)

    file_path = os.path.join(gexf_folder, f"{country_name}.gexf")

    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return False

    try:

        G = nx.read_gexf(file_path)
        print(f"Loaded network for {country_name}")


        create_final_intra_group_heatmap(G, country_name, output_folder)
        return True

    except Exception as e:
        print(f"Error processing {country_name}: {e}")
        return False


def create_final_intra_group_heatmap(G, country, output_folder):

    node_groups = nx.get_node_attributes(G, 'group')
    n_groups = 16


    group_adjacency = np.zeros((n_groups, n_groups))

    for edge in G.edges():
        group_i = int(node_groups[edge[0]])
        group_j = int(node_groups[edge[1]])
        group_adjacency[group_i, group_j] += 1
        if group_i != group_j:
            group_adjacency[group_j, group_i] += 1

    plt.figure(figsize=(12, 10))

    age_groups = [f"Age {i}" for i in range(n_groups)]
    heatmap = sns.heatmap(group_adjacency,
                          annot=True,
                          fmt='.0f',
                          cmap='Blues',
                          xticklabels=age_groups,
                          yticklabels=age_groups,
                          cbar_kws={'label': 'Connection Count'},
                          annot_kws={"size": 15},
                          square=True,
)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    cbar.set_label('Connection Count', size=22)


    plt.title(f'{country}', fontsize=30)
    plt.xlabel('Target Group', fontsize=22)
    plt.ylabel('Source Group', fontsize=22)

    plt.xticks(rotation=22, ha='right', fontsize=22)
    plt.yticks(fontsize=22)


    plt.tight_layout()

    heatmap_path = os.path.join(output_folder, f'{country}_final_group_heatmap.png')
    plt.savefig(heatmap_path, dpi=450, bbox_inches='tight')
    plt.close()



    plt.figure(figsize=(10, 7))

    intra_group_connections = np.diag(group_adjacency)

    bars = plt.bar(age_groups, intra_group_connections,
                   color=plt.cm.Greens(intra_group_connections / np.max(intra_group_connections)),
                   alpha=0.8)

    for bar, value in zip(bars, intra_group_connections):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{int(value)}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Age Groups', fontsize=12)
    plt.ylabel('Intra-Group Connections', fontsize=12)
    plt.title(f'Intra-Group Strength - {country}', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()


    output_path = os.path.join(output_folder, f'{country}_final_intra_group_bars.png')
    plt.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved final intra-group bar chart: {output_path}")


def main():
    countries_to_process = [
        "Germany", "Uganda", "Qatar", "Monaco"
    ]

    print("Generating final enhanced heatmaps with optimal cell sizes...")

    success_count = 0
    for country in countries_to_process:
        print(f"Processing {country}...")
        if load_and_generate_intra_group_heatmap(country):
            success_count += 1




if __name__ == "__main__":
    main()
