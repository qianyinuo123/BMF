import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


gexf_folder = "gexf_networks"
output_folder = "chord_diagrams"


os.makedirs(output_folder, exist_ok=True)


SELECTED_COUNTRIES = ["Germany", "Uganda", "Qatar", "Monaco"]


def plot_chord_diagram(G, country, output_folder):
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


    fig, ax = plt.subplots(figsize=(12, 10))


    radius = 0.8
    angles = np.linspace(0, 2 * np.pi, n_groups, endpoint=False)
    positions = np.array([(radius * np.cos(angle), radius * np.sin(angle))
                          for angle in angles])


    group_colors = plt.cm.tab20(np.arange(n_groups) % 20)
    connection_colors = plt.cm.viridis


    connections_to_draw = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            strength = normalized_matrix[i, j]
            if strength > 0.01:  # 只绘制显著连接
                connections_to_draw.append((i, j, strength))


                verts = [positions[i], (0, 0), positions[j]]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(verts, codes)


                linewidth = max(0.5, strength * 12)
                color = connection_colors(strength)

                patch = patches.PathPatch(path,
                                          facecolor='none',
                                          edgecolor=color,
                                          alpha=0.7,
                                          linewidth=linewidth)
                ax.add_patch(patch)


    group_sizes = {}
    for group in node_groups.values():
        group_sizes[group] = group_sizes.get(group, 0) + 1


    for i, (angle, pos) in enumerate(zip(angles, positions)):
        size = group_sizes.get(i, 0)


        node_size = 60 + (size / 150)


        ax.scatter(pos[0], pos[1],
                   s=node_size,
                   color=group_colors[i],
                   alpha=0.9,
                   edgecolors='black',
                   linewidth=1.5,
                   zorder=10)


        label_angle = np.degrees(angle)
        ha = 'right' if 90 < label_angle < 270 else 'left'
        label_pos = (pos[0] * 1.1, pos[1] * 1.1)

        ax.text(label_pos[0], label_pos[1], f'Age {i}',
                ha=ha, va='center',
                fontsize=14,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor='white',
                          alpha=0.8))

        if size > 0:
            ax.text(pos[0], pos[1], f'{size:,}',
                    ha='center', va='center',
                    fontsize=12,
                    fontweight='bold',
                    color='white')


    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')


    title = f'Age Group Connectivity - {country}'
    subtitle = f'Total: {len(G.nodes()):,} people, {len(G.edges()):,} connections'
    ax.set_title(f'{title}\n{subtitle}',
                 fontsize=18,
                 fontweight='bold',
                 pad=10)


    legend_text = "Circles: Age groups (size = group size)\nLines: Connections (thickness = strength)"
    ax.text(-1.05, -1.05, legend_text,
            fontsize=14,
            ha='left', va='bottom',
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor='lightyellow',
                      alpha=0.8))


    output_path = os.path.join(output_folder, f'{country}_chord_diagram.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
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
            print(f"📊 {country}: {len(G.nodes()):,} nodes, {len(G.edges()):,} edges")

            if plot_chord_diagram(G, country, output_folder):
                success_count += 1

        except Exception as e:
            print(f"Error processing {country}: {e}")

    print("=" * 50)
    print(f"🎉 Generated {success_count} chord diagrams")
    print(f"📁 Saved in: {output_folder}")
    print("=" * 50)


if __name__ == "__main__":
    main()
