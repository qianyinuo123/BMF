import os
import numpy as np
import pandas as pd
import networkx as nx
import pickle

# ========== PARAMETERS ==========
population_folder = "population fraction"
matrices_folder = "processed_matrices"
output_folder = "gexf_networks"
N = 10000

os.makedirs(output_folder, exist_ok=True)

group_names = list(range(16))
print(group_names)


def symmetrize_contact_matrix(matrix, counts):
    """
    Symmetrize contact matrix using formula: Cij_new = (Ni*Cij + Nj*Cji) / (2*Ni)

    Args:
        matrix (numpy.ndarray): original contact matrix
        counts (list): node counts for each age group

    Returns:
        numpy.ndarray: symmetrized matrix
    """
    n_groups = len(counts)
    symmetrized_matrix = np.zeros_like(matrix)

    for i in range(n_groups):
        for j in range(n_groups):
            Ni = counts[i]
            Nj = counts[j]
            if Ni > 0:
                symmetrized_matrix[i, j] = (Ni * matrix[i, j] + Nj * matrix[j, i]) / (2 * Ni)

    return symmetrized_matrix


def process_contact_matrices():
    """
    Process all contact matrix files, symmetrize and save new files.
    """
    print("Processing contact matrices...")

    for file_name in os.listdir(matrices_folder):
        if not file_name.endswith('.pkl'):
            continue

        country = file_name.replace('.pkl', '')
        if country == file_name:
            continue

        input_path = os.path.join(matrices_folder, file_name)
        output_path = os.path.join(matrices_folder, f"{country}-symmetrized.pkl")

        try:
            with open(input_path, 'rb') as f:
                original_matrix = pickle.load(f)

            population_file = os.path.join(population_folder, f"{country}-2023.xlsx")
            if not os.path.exists(population_file):
                print(f"Population file for {country} not found, skipping")
                continue

            df = pd.read_excel(population_file, usecols=[4])
            ratios = df.iloc[:16, 0].to_numpy(dtype=float)

            counts = np.round(N * ratios).astype(int)
            diff = N - counts.sum()
            if diff != 0:
                counts[np.argmax(counts)] += diff

            symmetrized_matrix = symmetrize_contact_matrix(original_matrix, counts)

            with open(output_path, 'wb') as f:
                pickle.dump(symmetrized_matrix, f)

            print(f"Processed contact matrix for {country}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")


def build_network_edges(G, symmetrized_matrix, group_nodes_dict, country):
    """
    Build network edges according to symmetrized matrix (Poisson distribution version).

    Args:
        G (networkx.Graph): graph object
        symmetrized_matrix (numpy.ndarray): symmetrized contact matrix
        group_nodes_dict (dict): dictionary of node lists per group
        country (str): country name
    """
    n_groups = len(group_nodes_dict)

    np.random.seed(42)

    connection_params = []

    # Intra-group connections
    for i in range(n_groups):
        nodes_in_group = group_nodes_dict[i]
        n_i = len(nodes_in_group)

        if n_i < 2:
            continue

        avg_degree = symmetrized_matrix[i, i]

        if n_i > 1:
            p = avg_degree / (n_i - 1)
        else:
            p = 0

        p = max(0, min(1, p))

        print(f"Intra-group {i}: n={n_i}, target avg deg={avg_degree:.3f}, p={p:.4f}")

        connection_params.append({
            'country': country,
            'group_i': i,
            'group_j': i,
            'connection_type': 'intra_group',
            'n_i': n_i,
            'n_j': n_i,
            'target_avg_degree': avg_degree,
            'probability_p': p,
            'max_possible_edges': n_i * (n_i - 1) // 2 if n_i > 1 else 0
        })

        for idx_i in range(n_i):
            for idx_j in range(idx_i + 1, n_i):
                if np.random.random() < p:
                    src = nodes_in_group[idx_i]
                    tgt = nodes_in_group[idx_j]
                    G.add_edge(src, tgt)

    # Inter-group connections
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            nodes_i = group_nodes_dict[i]
            nodes_j = group_nodes_dict[j]
            n_i = len(nodes_i)
            n_j = len(nodes_j)

            if n_i == 0 or n_j == 0:
                continue

            avg_degree_ij = symmetrized_matrix[i, j]

            p = avg_degree_ij / n_j

            p = max(0, min(1, p))

            print(f"Inter-group {i}-{j}: n_i={n_i}, n_j={n_j}, target avg deg={avg_degree_ij:.3f}, p={p:.4f}")

            connection_params.append({
                'country': country,
                'group_i': i,
                'group_j': j,
                'connection_type': 'inter_group',
                'n_i': n_i,
                'n_j': n_j,
                'target_avg_degree': avg_degree_ij,
                'probability_p': p,
                'max_possible_edges': n_i * n_j
            })

            for src in nodes_i:
                for tgt in nodes_j:
                    if np.random.random() < p:
                        G.add_edge(src, tgt)

    return connection_params


def save_connection_params(connection_params, country):
    """
    Save connection parameters to CSV file.

    Args:
        connection_params (list): list of connection parameter dicts
        country (str): country name
    """
    if not connection_params:
        return

    df = pd.DataFrame(connection_params)

    column_order = [
        'country', 'group_i', 'group_j', 'connection_type',
        'n_i', 'n_j', 'target_avg_degree', 'probability_p', 'max_possible_edges'
    ]
    df = df[column_order]

    csv_filename = f"{country}_connection_parameters.csv"
    csv_path = os.path.join(output_folder, csv_filename)
    df.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"Saved connection parameters for {country} to: {csv_path}")


def construct_networks():
    """
    Construct networks for all countries.
    """
    print("Constructing networks...")

    generated_countries = []
    missing_data_info = []

    all_countries_params = []

    for file_name in os.listdir(population_folder):
        if not file_name.endswith(".xlsx"):
            continue

        country = file_name.replace("-2023.xlsx", "")
        population_file_path = os.path.join(population_folder, file_name)

        matrix_file_path = os.path.join(matrices_folder, f"{country}-symmetrized.pkl")
        if not os.path.exists(matrix_file_path):
            print(f"Symmetrized matrix for {country} not found, skipping")
            missing_data_info.append({
                'country': country,
                'missing_file': f"{country}-symmetrized.pkl",
                'file_type': 'symmetrized matrix file',
                'expected_path': matrix_file_path
            })
            continue

        try:
            df = pd.read_excel(population_file_path, usecols=[4])
            ratios = df.iloc[:16, 0].to_numpy(dtype=float)

            if not np.isclose(ratios.sum(), 1.0, atol=1e-3):
                print(f"{country}: ratio sum = {ratios.sum():.4f}")

            counts = np.round(N * ratios).astype(int)
            diff = N - counts.sum()
            if diff != 0:
                counts[15] += diff

            node_ids = np.arange(N)
            group_list = []
            for name, cnt in zip(group_names, counts):
                group_list.extend([name] * cnt)
            group_list = group_list[:N]

            G = nx.Graph()
            for i in range(N):
                G.add_node(int(node_ids[i]), group=group_list[i])

            group_nodes_dict = {}
            for i in range(N):
                group = group_list[i]
                if group not in group_nodes_dict:
                    group_nodes_dict[group] = []
                group_nodes_dict[group].append(int(node_ids[i]))

            with open(matrix_file_path, 'rb') as f:
                symmetrized_matrix = pickle.load(f)

            connection_params = build_network_edges(G, symmetrized_matrix, group_nodes_dict, country)

            save_connection_params(connection_params, country)

            all_countries_params.extend(connection_params)

            output_path = os.path.join(output_folder, f"{country}.gexf")
            nx.write_gexf(G, output_path)
            print(f"Processed {country}: {len(G.nodes())} nodes, {len(G.edges())} edges")

            generated_countries.append(country)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            missing_data_info.append({
                'country': country,
                'missing_file': 'Processing error',
                'file_type': 'unknown',
                'error': str(e),
                'expected_path': matrix_file_path
            })

    if all_countries_params:
        all_df = pd.DataFrame(all_countries_params)
        summary_csv_path = os.path.join(output_folder, "all_countries_connection_parameters.csv")
        all_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        print(f"Saved all countries connection parameters summary to: {summary_csv_path}")

    generated_countries_file = os.path.join(output_folder, "generated_countries.txt")
    with open(generated_countries_file, 'w', encoding='utf-8') as f:
        for country in generated_countries:
            f.write(f"{country}\n")
    print(f"Saved generated countries list to: {generated_countries_file}")

    if missing_data_info:
        missing_data_file = os.path.join(output_folder, "missing_data_details.csv")
        missing_df = pd.DataFrame(missing_data_info)
        missing_df.to_csv(missing_data_file, index=False, encoding='utf-8')
        print(f"Saved missing data details to: {missing_data_file}")

    print(f"\nNetwork generation statistics:")
    print(f"   Successfully generated countries: {len(generated_countries)}")
    print(f"   Countries with missing data: {len(missing_data_info)}")

    if missing_data_info:
        print("   Missing data details:")
        for info in missing_data_info:
            print(f"     - {info['country']}: missing file {info.get('missing_file', 'N/A')}")

    return generated_countries, missing_data_info


if __name__ == "__main__":
    process_contact_matrices()
    generated_countries, missing_data_info = construct_networks()
    print("All processing completed! GEXF files saved in 'gexf_networks-151——100/' folder.")