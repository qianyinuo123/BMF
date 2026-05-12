import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ========== PARAMETERS ==========
gexf_folder = ""
output_folder = "chord_diagrams-new"

os.makedirs(output_folder, exist_ok=True)

SELECTED_COUNTRIES = ["Germany", "Uganda", "Qatar", "Monaco"]


def plot_chord_diagram(G, country, output_folder):
    """Create chord diagram for a given country"""
    print(f"Creating chord diagram for {country}...")

    node_groups = nx.get_node_attributes(G, 'group')

    n_groups = 16
    connection_matrix = np.zeros((n_groups, n_groups))

    for edge in G.edges():
        i = int(node_groups[edge[0]])
        j = int(node_groups[edge[1]])
        connection_matrix[i, j] += 1
        if i != j:
            connection_matrix[j, i] += 1

    max_connections = np.max(connection_matrix)
    if max_connections > 0:
        normalized_matrix = connection_matrix / max_connections
    else:
        normalized_matrix = connection_matrix

    group_sizes = {}
    for group in node_groups.values():
        group_sizes[group] = group_sizes.get(group, 0) + 1

    min_size = min(group_sizes.values())
    max_size = max(group_sizes.values())

    def get_node_size(group_id):
        size = group_sizes.get(group_id, 0)
        if max_size == min_size:
            return 120
        node_size = 60 + (size - min_size) / (max_size - min_size) * 700
        return node_size

    fig, ax = plt.subplots(figsize=(10, 10))

    radius = 0.85
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False)
    positions = np.array([(radius * np.cos(angle), radius * np.sin(angle))
                          for angle in angles])

    group_colors = plt.cm.tab20(np.arange(n_groups) % 20)
    connection_colors = plt.cm.viridis

    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            strength = normalized_matrix[i, j]
            if strength > 0.01:
                verts = [positions[i], (0, 0), positions[j]]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(verts, codes)

                linewidth = max(0.5, strength * 15)
                color = connection_colors(strength)

                patch = patches.PathPatch(path,
                                          facecolor='none',
                                          edgecolor=color,
                                          alpha=0.7,
                                          linewidth=linewidth)
                ax.add_patch(patch)

    for i, (angle, pos) in enumerate(zip(angles, positions)):
        node_size = get_node_size(i)

        ax.scatter(pos[0], pos[1],
                   s=node_size,
                   color=group_colors[i],
                   alpha=0.9,
                   edgecolors='black',
                   linewidth=2,
                   zorder=10)

        label_angle = np.degrees(angle)
        label_offset = 1.25
        label_pos = (pos[0] * label_offset, pos[1] * label_offset)

        if -45 <= label_angle <= 45:
            ha, va = 'center', 'bottom'
        elif 45 < label_angle <= 135:
            ha, va = 'left', 'center'
        elif 135 < label_angle <= 225:
            ha, va = 'center', 'top'
        else:
            ha, va = 'right', 'center'

        ax.text(label_pos[0], label_pos[1], f'{i}',
                ha=ha, va=va,
                fontsize=30,
                color='black')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    title = f'{country}'
    ax.set_title(title,
                 fontsize=40,
                 pad=0.8)

    output_path = os.path.join(output_folder, f'{country}_chord_diagram.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"   Group sizes: {group_sizes}")
    print(
        f"   Node size range: {min_size} -> {get_node_size(list(group_sizes.keys())[0]):.1f}px to {max_size} -> {get_node_size(list(group_sizes.keys())[-1]):.1f}px")

    return True


def main():
    print("=" * 50)
    print("CHORD DIAGRAM GENERATOR")
    print("=" * 50)

    success_count = 0

    for country in SELECTED_COUNTRIES:
        file_path = os.path.join(gexf_folder, f"{country}.gexf")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            G = nx.read_gexf(file_path)
            print(f"{country}: {len(G.nodes()):,} nodes, {len(G.edges()):,} edges")

            if plot_chord_diagram(G, country, output_folder):
                success_count += 1

        except Exception as e:
            print(f"Error processing {country}: {e}")

    print("=" * 50)
    print(f"Generated {success_count} chord diagrams")
    print(f"Saved in: {output_folder}")
    print("=" * 50)


if __name__ == "__main__":
    main()