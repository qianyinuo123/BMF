import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class AgeStructuredSIRModel:

    def __init__(self, gamma=0.1, max_steps=200):
        self.gamma = gamma
        self.max_steps = max_steps

    def load_network(self, country, networks_folder):
        network_path = os.path.join(networks_folder, f"{country}.gexf")

        if not os.path.exists(network_path):
            print(f"Network file not found: {network_path}")
            return None

        try:
            G = nx.read_gexf(network_path)
            G = nx.convert_node_labels_to_integers(G)
            print(f"Loaded {country} network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
        except Exception as e:
            print(f"Failed to load network: {e}")
            return None

    def extract_age_groups(self, G):
        age_groups = {}

        for node in G.nodes():
            try:
                group_value = G.nodes[node].get('group', 0)

                if isinstance(group_value, (int, np.integer)):
                    age_groups[node] = int(group_value)
                else:
                    try:
                        age_groups[node] = int(float(group_value))
                    except:
                        age_groups[node] = 0
            except:
                age_groups[node] = 0

        return age_groups

    def discrete_time_sir_simulation(self, G, beta, initial_infected_fraction=0.01):
        N = G.number_of_nodes()

        status = np.zeros(N, dtype=int)

        n_initial_infected = max(1, int(N * initial_infected_fraction))
        initial_nodes = np.random.choice(N, n_initial_infected, replace=False)
        status[initial_nodes] = 1

        age_groups = self.extract_age_groups(G)

        S_series = [N - n_initial_infected]
        I_series = [n_initial_infected]
        R_series = [0]

        n_age_groups = 16
        age_S_series = {i: [] for i in range(n_age_groups)}
        age_I_series = {i: [] for i in range(n_age_groups)}
        age_R_series = {i: [] for i in range(n_age_groups)}

        for age in range(n_age_groups):
            age_nodes = [node for node, ag in age_groups.items() if ag == age]
            if age_nodes:
                age_S_series[age].append(sum(1 for node in age_nodes if status[node] == 0))
                age_I_series[age].append(sum(1 for node in age_nodes if status[node] == 1))
                age_R_series[age].append(sum(1 for node in age_nodes if status[node] == 2))
            else:
                age_S_series[age].append(0)
                age_I_series[age].append(0)
                age_R_series[age].append(0)

        adjacency = [list(G.neighbors(node)) for node in range(N)]

        for step in range(self.max_steps):
            new_status = status.copy()
            infected_nodes = np.where(status == 1)[0]

            for node in infected_nodes:
                neighbors = adjacency[node]
                for neighbor in neighbors:
                    if status[neighbor] == 0:
                        if np.random.random() < beta:
                            new_status[neighbor] = 1

            for node in infected_nodes:
                if np.random.random() < self.gamma:
                    new_status[node] = 2

            status = new_status

            S_series.append(np.sum(status == 0))
            I_series.append(np.sum(status == 1))
            R_series.append(np.sum(status == 2))

            for age in range(n_age_groups):
                age_nodes = [node for node, ag in age_groups.items() if ag == age]
                if age_nodes:
                    age_S_series[age].append(sum(1 for node in age_nodes if status[node] == 0))
                    age_I_series[age].append(sum(1 for node in age_nodes if status[node] == 1))
                    age_R_series[age].append(sum(1 for node in age_nodes if status[node] == 2))
                else:
                    age_S_series[age].append(0)
                    age_I_series[age].append(0)
                    age_R_series[age].append(0)

            if I_series[-1] == 0:
                print(f"Simulation ended at step {step + 1} (no infected nodes)")
                break

        final_S = S_series[-1]
        final_I = I_series[-1]
        final_R = R_series[-1]
        final_infected_ratio = final_R / N

        final_age_stats = {}
        for age in range(n_age_groups):
            age_nodes = [node for node, ag in age_groups.items() if ag == age]
            total_in_age = len(age_nodes)
            if total_in_age > 0:
                final_age_stats[age] = {
                    'S_final': sum(1 for node in age_nodes if status[node] == 0) / total_in_age,
                    'I_final': sum(1 for node in age_nodes if status[node] == 1) / total_in_age,
                    'R_final': sum(1 for node in age_nodes if status[node] == 2) / total_in_age,
                    'total': total_in_age
                }
            else:
                final_age_stats[age] = {
                    'S_final': 0, 'I_final': 0, 'R_final': 0, 'total': 0
                }

        R_all_weighted = 0.0
        for age in range(n_age_groups):
            age_total = final_age_stats[age]['total']
            if age_total > 0:
                R_all_weighted += final_age_stats[age]['R_final'] * (age_total / N)

        return {
            'final_S': final_S,
            'final_I': final_I,
            'final_R': final_R,
            'final_infected_ratio': final_infected_ratio,
            'R_all_weighted': R_all_weighted,
            'S_series': S_series,
            'I_series': I_series,
            'R_series': R_series,
            'age_S_series': age_S_series,
            'age_I_series': age_I_series,
            'age_R_series': age_R_series,
            'final_age_stats': final_age_stats,
            'total_steps': len(S_series) - 1,
            'age_groups': age_groups
        }

    def run_single_country_analysis(self, country, networks_folder, beta_values,
                                    n_simulations=10, initial_infected_fraction=0.01):
        print(f"\n{'=' * 60}")
        print(f"Starting analysis: {country}")
        print(f"{'=' * 60}")

        G = self.load_network(country, networks_folder)
        if G is None:
            return None

        N = G.number_of_nodes()
        avg_degree = np.mean([d for n, d in G.degree()])
        print(f"Network info: {N} nodes, average degree: {avg_degree:.2f}")

        age_groups = self.extract_age_groups(G)
        age_distribution = {}
        for age in range(16):
            count = sum(1 for ag in age_groups.values() if ag == age)
            if count > 0:
                age_distribution[age] = count

        print(f"Age distribution: {age_distribution}")

        results = {
            'country': country,
            'network_info': {
                'N': N,
                'avg_degree': avg_degree,
                'age_distribution': age_distribution
            },
            'beta_results': []
        }

        for beta in tqdm(beta_values, desc=f"Scanning Beta - {country}"):
            beta_results = {
                'beta': beta,
                'R0': beta / self.gamma,
                'simulations': []
            }

            infected_ratios = []
            R_all_values = []

            for sim in range(n_simulations):
                sim_result = self.discrete_time_sir_simulation(
                    G, beta, initial_infected_fraction
                )

                beta_results['simulations'].append(sim_result)
                infected_ratios.append(sim_result['final_infected_ratio'])
                R_all_values.append(sim_result['R_all_weighted'])

            beta_results['mean_infected_ratio'] = np.mean(infected_ratios)
            beta_results['std_infected_ratio'] = np.std(infected_ratios)
            beta_results['min_infected_ratio'] = np.min(infected_ratios)
            beta_results['max_infected_ratio'] = np.max(infected_ratios)
            beta_results['median_infected_ratio'] = np.median(infected_ratios)
            beta_results['mean_R_all'] = np.mean(R_all_values)

            n_age_groups = 16
            age_infected_means = {}
            for age in range(n_age_groups):
                age_infections = []
                for sim_result in beta_results['simulations']:
                    if age in sim_result['final_age_stats']:
                        age_infections.append(sim_result['final_age_stats'][age]['R_final'])
                if age_infections:
                    age_infected_means[age] = np.mean(age_infections)
                else:
                    age_infected_means[age] = 0.0

            beta_results['age_infected_means'] = age_infected_means

            results['beta_results'].append(beta_results)

            print(f"  Beta={beta:.3f}: mean_infected_ratio={beta_results['mean_infected_ratio']:.4f} "
                  f"(std={beta_results['std_infected_ratio']:.4f})")

        return results

    def run_multiple_countries(self, countries, networks_folder, beta_values,
                               n_simulations=10, output_folder="results"):
        os.makedirs(output_folder, exist_ok=True)

        all_results = {}

        for country in countries:
            print(f"\n{'=' * 60}")
            print(f"Processing country: {country}")
            print(f"{'=' * 60}")

            country_results = self.run_single_country_analysis(
                country, networks_folder, beta_values, n_simulations
            )

            if country_results is not None:
                all_results[country] = country_results

                self.save_country_results(country_results, output_folder)

                self.generate_country_report(country_results, output_folder)

        if len(all_results) > 1:
            self.generate_comparison_report(all_results, output_folder)

        return all_results

    def save_country_results(self, country_results, output_folder):
        country = country_results['country']
        beta_results = country_results['beta_results']

        data_rows = []
        for br in beta_results:
            row = {
                'Beta': br['beta'],
                'R0': br['R0'],
                'mean_infected_ratio': br['mean_infected_ratio'],
                'std_infected_ratio': br['std_infected_ratio'],
                'mean_R_all': br['mean_R_all']
            }

            for age in range(16):
                row[f'R_age{age + 1}'] = br['age_infected_means'].get(age, 0.0)

            data_rows.append(row)

        df = pd.DataFrame(data_rows)

        csv_file = os.path.join(output_folder, f"Discrete_SIR_{country}.csv")
        df.to_csv(csv_file, index=False)
        print(f"Results saved: {csv_file}")

        return csv_file

    def generate_country_report(self, country_results, output_folder):
        country = country_results['country']
        beta_results = country_results['beta_results']
        network_info = country_results['network_info']

        betas = [br['beta'] for br in beta_results]
        mean_infected = [br['mean_infected_ratio'] for br in beta_results]
        std_infected = [br['std_infected_ratio'] for br in beta_results]

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(betas, mean_infected, 'b-', linewidth=2, label='Mean Infection Ratio')
        plt.fill_between(betas,
                         np.array(mean_infected) - np.array(std_infected),
                         np.array(mean_infected) + np.array(std_infected),
                         alpha=0.2, color='blue', label='±1 Standard Deviation')

        beta_threshold = None
        for i, mean_val in enumerate(mean_infected):
            if mean_val > 0.01:
                beta_threshold = betas[i]
                plt.axvline(x=beta_threshold, color='r', linestyle='--',
                            label=f'Threshold ≈ {beta_threshold:.3f}')
                break

        plt.xlabel('Transmission Rate β', fontsize=12)
        plt.ylabel('Final Infection Ratio R(∞)/N', fontsize=12)
        plt.title(f'{country} - SIR Model Transmission (γ={self.gamma})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 2, 2)
        mid_idx = len(beta_results) // 2
        sample_sim = beta_results[mid_idx]['simulations'][0]

        time_steps = range(len(sample_sim['S_series']))
        plt.plot(time_steps, sample_sim['S_series'], 'g-', label='Susceptible S(t)')
        plt.plot(time_steps, sample_sim['I_series'], 'r-', label='Infected I(t)')
        plt.plot(time_steps, sample_sim['R_series'], 'b-', label='Recovered R(t)')

        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Number of Nodes', fontsize=12)
        plt.title(f'Time Series Example (β={betas[mid_idx]:.3f})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)

        n_betas = len(betas)
        n_age_groups = 16

        age_infection_matrix = np.zeros((n_age_groups, n_betas))
        for i, br in enumerate(beta_results):
            for age in range(n_age_groups):
                age_infection_matrix[age, i] = br['age_infected_means'].get(age, 0.0)

        im = plt.imshow(age_infection_matrix, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im, label='Infection Ratio')
        plt.xlabel('Beta Index', fontsize=12)
        plt.ylabel('Age Group', fontsize=12)
        plt.title('Infection Ratio Heatmap by Age Groups', fontsize=14)

        plt.yticks(range(n_age_groups), [f'Age {i + 1}' for i in range(n_age_groups)])

        plt.subplot(2, 2, 4)

        networks_folder = "."
        G = self.load_network(country, networks_folder)
        if G is not None:
            degrees = [d for n, d in G.degree()]
            plt.hist(degrees, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Node Degree', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Network Degree Distribution (Avg. Degree={network_info["avg_degree"]:.2f})', fontsize=14)
            plt.grid(True, alpha=0.3)

    def generate_comparison_report(self, all_results, output_folder):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)

        for country, results in all_results.items():
            betas = [br['beta'] for br in results['beta_results']]
            mean_infected = [br['mean_infected_ratio'] for br in results['beta_results']]
            plt.plot(betas, mean_infected, '-', linewidth=2, label=country)

        plt.xlabel('Transmission Rate β', fontsize=12)
        plt.ylabel('Final Infection Ratio R(∞)/N', fontsize=12)
        plt.title(f'Multi-Country SIR Transmission Comparison (γ={self.gamma})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 2, 2)

        countries = list(all_results.keys())
        n_countries = len(countries)

        age_distributions = []
        for country, results in all_results.items():
            age_dist = results['network_info']['age_distribution']
            total_nodes = results['network_info']['N']
            age_props = [age_dist.get(i, 0) / total_nodes for i in range(16)]
            age_distributions.append(age_props)

        bar_width = 0.8 / n_countries
        x_pos = np.arange(16)

        for i, (country, age_props) in enumerate(zip(countries, age_distributions)):
            offset = (i - n_countries / 2 + 0.5) * bar_width
            plt.bar(x_pos + offset, age_props, width=bar_width, label=country)

        plt.xlabel('Age Group', fontsize=12)
        plt.ylabel('Proportion', fontsize=12)
        plt.title('Age Group Distribution Comparison', fontsize=14)
        plt.xticks(x_pos, [f'Age {i + 1}' for i in range(16)], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        plt.subplot(2, 2, 3)

        network_props = []
        for country, results in all_results.items():
            props = results['network_info']
            network_props.append({
                'country': country,
                'N': props['N'],
                'avg_degree': props['avg_degree']
            })

        cell_text = []
        for prop in network_props:
            cell_text.append([
                prop['country'],
                f"{prop['N']:,}",
                f"{prop['avg_degree']:.2f}"
            ])

        plt.table(cellText=cell_text,
                  colLabels=['Country', 'Nodes', 'Average Degree'],
                  cellLoc='center',
                  loc='center')
        plt.axis('off')
        plt.title('Network Properties Comparison', fontsize=14)

        plt.subplot(2, 2, 4)

        for country, results in all_results.items():
            betas = [br['beta'] for br in results['beta_results']]
            mean_infected = [br['mean_infected_ratio'] for br in results['beta_results']]

            non_zero_idx = [i for i, val in enumerate(mean_infected) if val > 0.001]
            if len(non_zero_idx) > 3:
                plt.loglog([betas[i] for i in non_zero_idx],
                           [mean_infected[i] for i in non_zero_idx],
                           'o-', label=country)

        plt.xlabel('Transmission Rate β (Log Scale)', fontsize=12)
        plt.ylabel('Infection Ratio R(∞)/N (Log Scale)', fontsize=12)
        plt.title('Critical Behavior Comparison', fontsize=14)
        plt.grid(True, alpha=0.3, which='both')
        plt.legend()


def main():
    networks_folder = "gexf_networks_no_school"
    output_folder = "Discrete_SIR_Analysis_Results-0.001-no_school"

    selected_countries = [
        "Uganda",
        "Qatar",
        "Monaco",
        "Germany",
    ]

    gamma = 0.333333333333333333
    initial_infected_fraction = 0.001
    max_steps = 200

    beta_min = 0.0
    beta_max = 0.4
    num_beta = 401
    beta_values = np.linspace(beta_min, beta_max, num_beta)

    n_simulations = 10

    print("=" * 70)
    print("AGE-STRUCTURED SIR MODEL ANALYSIS (Discrete-time Synchronous Update)")
    print("=" * 70)
    print(f"Recovery rate gamma: {gamma}")
    print(f"Initial infection fraction: {initial_infected_fraction}")
    print(f"Maximum time steps: {max_steps}")
    print(f"Beta range: {beta_min} to {beta_max} ({num_beta} values)")
    print(f"Simulations per Beta: {n_simulations}")
    print(f"Countries analyzed: {', '.join(selected_countries)}")
    print(f"Network folder: {networks_folder}")
    print(f"Output folder: {output_folder}")
    print("=" * 70)

    model = AgeStructuredSIRModel(gamma=gamma, max_steps=max_steps)

    try:
        all_results = model.run_multiple_countries(
            countries=selected_countries,
            networks_folder=networks_folder,
            beta_values=beta_values,
            n_simulations=n_simulations,
            output_folder=output_folder
        )

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETED!")
        print("=" * 70)

        if all_results:
            print("\nSummary statistics:")
            print("-" * 70)
            print(f"{'Country':<10} {'Nodes':<10} {'Avg Degree':<10} {'Threshold(β)':<15}")
            print("-" * 70)

            for country, results in all_results.items():
                beta_results = results['beta_results']
                mean_infected = [br['mean_infected_ratio'] for br in beta_results]
                betas = [br['beta'] for br in beta_results]

                beta_threshold = None
                for i, mean_val in enumerate(mean_infected):
                    if mean_val > 0.01:
                        beta_threshold = betas[i]
                        break

                net_info = results['network_info']
                print(f"{country:<10} {net_info['N']:<10,} {net_info['avg_degree']:<10.2f} "
                      f"{beta_threshold if beta_threshold else 'N/A':<15.4f}")

        print(f"\nAll results saved to folder: {output_folder}")

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
