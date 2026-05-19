import numpy as np
import pandas as pd
import networkx as nx
import os
from tqdm import tqdm
import warnings
import concurrent.futures as cf
import multiprocessing
import time
import random

warnings.filterwarnings('ignore')


def simulate_one_beta(model, G, beta, n_simulations, initial_infected_fraction):
    seed = (os.getpid() * int(time.time())) % (2**32)
    np.random.seed(seed)
    random.seed(seed)

    beta_results = {
        'beta': beta,
        'R0': beta / model.gamma,
        'simulations': []
    }

    infected_ratios = []
    R_all_values = []

    for _ in range(n_simulations):
        sim_result = model.gillespie_sir_simulation(
            G, beta, initial_infected_fraction
        )
        beta_results['simulations'].append(sim_result)
        infected_ratios.append(sim_result['final_infected_ratio'])
        R_all_values.append(sim_result['R_all_weighted'])

    mean_val = np.mean(infected_ratios)
    sd_val = np.std(infected_ratios, ddof=1)
    ci95 = 1.96 * sd_val / np.sqrt(len(infected_ratios))

    beta_results['mean_infected_ratio'] = mean_val
    beta_results['std_infected_ratio'] = sd_val
    beta_results['sd_infected_ratio'] = sd_val
    beta_results['ci95_low'] = mean_val - ci95
    beta_results['ci95_high'] = mean_val + ci95
    beta_results['min_infected_ratio'] = np.min(infected_ratios)
    beta_results['max_infected_ratio'] = np.max(infected_ratios)
    beta_results['median_infected_ratio'] = np.median(infected_ratios)
    beta_results['mean_R_all'] = np.mean(R_all_values)

    age_infected_means = {}
    age_infected_sds = {}
    age_infected_ci95_low = {}
    age_infected_ci95_high = {}

    for age in range(16):
        age_infections = []
        for sim_result in beta_results['simulations']:
            if age in sim_result['final_age_stats']:
                age_infections.append(
                    sim_result['final_age_stats'][age]['R_final']
                )
        if len(age_infections) > 0:
            mean_age = np.mean(age_infections)
            sd_age = np.std(age_infections, ddof=1)
            ci95_age = 1.96 * sd_age / np.sqrt(len(age_infections))
            age_infected_means[age] = mean_age
            age_infected_sds[age] = sd_age
            age_infected_ci95_low[age] = mean_age - ci95_age
            age_infected_ci95_high[age] = mean_age + ci95_age
        else:
            age_infected_means[age] = 0.0
            age_infected_sds[age] = 0.0
            age_infected_ci95_low[age] = 0.0
            age_infected_ci95_high[age] = 0.0

    beta_results['age_infected_means'] = age_infected_means
    beta_results['age_infected_sds'] = age_infected_sds
    beta_results['age_infected_ci95_low'] = age_infected_ci95_low
    beta_results['age_infected_ci95_high'] = age_infected_ci95_high

    group_names = ["0-19", "20-39", "40-59", "60+"]
    four_group_summary = {}
    for gname in group_names:
        values = [
            sim['four_group_stats'][gname]['R_final']
            for sim in beta_results['simulations']
        ]
        mean_g = np.mean(values)
        sd_g = np.std(values, ddof=1)
        ci95_g = 1.96 * sd_g / np.sqrt(len(values))
        four_group_summary[gname] = {
            'mean': mean_g,
            'sd': sd_g,
            'ci95_low': mean_g - ci95_g,
            'ci95_high': mean_g + ci95_g
        }
    beta_results['four_group_summary'] = four_group_summary

    return beta_results


class AgeStructuredSIRModel:

    def __init__(self, gamma=0.1, max_time=200):
        self.gamma = gamma
        self.max_time = max_time

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

    def gillespie_sir_simulation(self, G, beta, initial_infected_fraction=0.001):
        N = G.number_of_nodes()
        status = np.zeros(N, dtype=np.int8)
        n_initial = max(1, int(N * initial_infected_fraction))
        initial_nodes = np.random.choice(N, n_initial, replace=False)
        status[initial_nodes] = 1

        age_groups = self.extract_age_groups(G)
        adjacency = [list(G.neighbors(i)) for i in range(N)]

        times = [0.0]
        S_series = [N - n_initial]
        I_series = [n_initial]
        R_series = [0]

        n_age_groups = 16
        age_S_series = {i: [] for i in range(n_age_groups)}
        age_I_series = {i: [] for i in range(n_age_groups)}
        age_R_series = {i: [] for i in range(n_age_groups)}

        def record_age_stats():
            for age in range(n_age_groups):
                age_nodes = [node for node, ag in age_groups.items() if ag == age]
                if len(age_nodes) == 0:
                    age_S_series[age].append(0)
                    age_I_series[age].append(0)
                    age_R_series[age].append(0)
                else:
                    age_S_series[age].append(np.sum(status[age_nodes] == 0))
                    age_I_series[age].append(np.sum(status[age_nodes] == 1))
                    age_R_series[age].append(np.sum(status[age_nodes] == 2))

        record_age_stats()
        current_time = 0.0

        while current_time < self.max_time:
            infected_nodes = np.where(status == 1)[0]
            if len(infected_nodes) == 0:
                break

            SI_edges = []
            for i in infected_nodes:
                for j in adjacency[i]:
                    if status[j] == 0:
                        SI_edges.append((i, j))

            infection_rate = beta * len(SI_edges)
            recovery_rate = self.gamma * len(infected_nodes)
            total_rate = infection_rate + recovery_rate
            if total_rate <= 0:
                break

            r1 = np.random.random()
            dt = -np.log(r1) / total_rate
            current_time += dt

            r2 = np.random.random() * total_rate
            if r2 < infection_rate:
                _, target = SI_edges[np.random.randint(len(SI_edges))]
                status[target] = 1
            else:
                recover_node = infected_nodes[np.random.randint(len(infected_nodes))]
                status[recover_node] = 2

            times.append(current_time)
            S_series.append(np.sum(status == 0))
            I_series.append(np.sum(status == 1))
            R_series.append(np.sum(status == 2))
            record_age_stats()

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
                    'S_final': np.sum(status[age_nodes] == 0) / total_in_age,
                    'I_final': np.sum(status[age_nodes] == 1) / total_in_age,
                    'R_final': np.sum(status[age_nodes] == 2) / total_in_age,
                    'total': total_in_age
                }
            else:
                final_age_stats[age] = {'S_final': 0, 'I_final': 0, 'R_final': 0, 'total': 0}

        four_age_groups = {
            "0-19": list(range(0, 4)),
            "20-39": list(range(4, 8)),
            "40-59": list(range(8, 12)),
            "60+": list(range(12, 16))
        }
        four_group_stats = {}
        for group_name, group_ids in four_age_groups.items():
            nodes = [node for node, ag in age_groups.items() if ag in group_ids]
            total = len(nodes)
            if total > 0:
                four_group_stats[group_name] = {
                    'R_final': np.sum(status[nodes] == 2) / total,
                    'total': total
                }
            else:
                four_group_stats[group_name] = {'R_final': 0, 'total': 0}

        R_all_weighted = 0.0
        for age in range(n_age_groups):
            age_total = final_age_stats[age]['total']
            if age_total > 0:
                R_all_weighted += final_age_stats[age]['R_final'] * (age_total / N)

        return {
            'times': times,
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
            'four_group_stats': four_group_stats,
            'total_steps': len(times),
            'age_groups': age_groups
        }

    def run_single_country_analysis(self, country, networks_folder, beta_values,
                                    n_simulations=50, initial_infected_fraction=0.001,
                                    max_workers=None):
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {country}")
        print(f"{'=' * 60}")

        G = self.load_network(country, networks_folder)
        if G is None:
            return None

        N = G.number_of_nodes()
        avg_degree = np.mean([d for n, d in G.degree()])
        print(f"Network info: {N} nodes, mean degree: {avg_degree:.2f}")

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

        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        print(f"Using {max_workers} processes for {len(beta_values)} beta values")

        args_list = [
            (self, G, beta, n_simulations, initial_infected_fraction)
            for beta in beta_values
        ]

        beta_results_list = []
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_beta = {
                executor.submit(simulate_one_beta, *args): args[2]
                for args in args_list
            }
            for future in tqdm(cf.as_completed(future_to_beta),
                               total=len(beta_values),
                               desc=f"Parallel simulation - {country}"):
                beta_results_list.append(future.result())

        beta_results_list.sort(key=lambda x: x['beta'])
        results['beta_results'] = beta_results_list

        for br in beta_results_list:
            print(f"  Beta={br['beta']:.3f}: Mean={br['mean_infected_ratio']:.4f}, "
                  f"SD={br['std_infected_ratio']:.4f}, "
                  f"95%CI=[{br['ci95_low']:.4f}, {br['ci95_high']:.4f}]")

        return results

    def run_multiple_countries(self, countries, networks_folder, beta_values,
                               n_simulations=50, output_folder="results",
                               max_workers=None):
        os.makedirs(output_folder, exist_ok=True)
        all_results = {}

        for country in countries:
            print(f"\n{'=' * 60}")
            print(f"Processing: {country}")
            print(f"{'=' * 60}")

            country_results = self.run_single_country_analysis(
                country, networks_folder, beta_values,
                n_simulations, initial_infected_fraction=0.001,
                max_workers=max_workers
            )
            if country_results is not None:
                all_results[country] = country_results
                self.save_country_results(country_results, output_folder)
                self.save_four_group_results(country_results, output_folder)

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
                'sd_infected_ratio': br['sd_infected_ratio'],
                'ci95_low': br['ci95_low'],
                'ci95_high': br['ci95_high'],
                'mean_R_all': br['mean_R_all']
            }
            for age in range(16):
                row[f'R_age{age+1}'] = br['age_infected_means'].get(age, 0.0)
                row[f'R_age{age+1}_sd'] = br['age_infected_sds'].get(age, 0.0)
                row[f'R_age{age+1}_ci95_low'] = br['age_infected_ci95_low'].get(age, 0.0)
                row[f'R_age{age+1}_ci95_high'] = br['age_infected_ci95_high'].get(age, 0.0)
            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        csv_file = os.path.join(output_folder, f"Gillespie_SIR_{country}.csv")
        df.to_csv(csv_file, index=False)
        print(f"Results saved: {csv_file}")
        return csv_file

    def save_four_group_results(self, country_results, output_folder):
        country = country_results['country']
        rows = []
        for br in country_results['beta_results']:
            sims = br['simulations']
            row = {'Beta': br['beta']}
            group_names = ["0-19", "20-39", "40-59", "60+"]
            for gname in group_names:
                values = [sim['four_group_stats'][gname]['R_final'] for sim in sims]
                mean_val = np.mean(values)
                sd_val = np.std(values, ddof=1)
                ci95 = 1.96 * sd_val / np.sqrt(len(values))
                row[f'{gname}_mean'] = mean_val
                row[f'{gname}_sd'] = sd_val
                row[f'{gname}_ci95_low'] = mean_val - ci95
                row[f'{gname}_ci95_high'] = mean_val + ci95
            rows.append(row)
        df = pd.DataFrame(rows)
        save_path = os.path.join(output_folder, f"Gillespie_4AgeGroups_{country}.csv")
        df.to_csv(save_path, index=False)
        print(f"Four age groups results saved: {save_path}")


def main():
    networks_folder = "gexf_networks_no_work"
    output_folder = "Gillespie_SIR_Results-no_school_fig.10(a)_and_fig.11"
    selected_countries = ["Uganda", "Qatar", "Monaco", "Germany"]

    gamma = 1 / 3
    initial_infected_fraction = 0.001
    max_time = 200

    beta_min = 0.0
    beta_max = 0.4
    num_beta = 101
    beta_values = np.linspace(beta_min, beta_max, num_beta)

    n_simulations = 30
    max_workers = multiprocessing.cpu_count()

    print("=" * 70)
    print("Age-structured Gillespie-SIR Model (Multi-process Parallel Version)")
    print("=" * 70)
    print(f"gamma = {gamma}")
    print(f"initial infected fraction = {initial_infected_fraction}")
    print(f"max_time = {max_time}")
    print(f"beta range = {beta_min} ~ {beta_max}")
    print(f"beta number = {num_beta}")
    print(f"independent realizations = {n_simulations}")
    print(f"countries = {selected_countries}")
    print(f"parallel processes = {max_workers}")
    print("=" * 70)

    model = AgeStructuredSIRModel(gamma=gamma, max_time=max_time)

    try:
        all_results = model.run_multiple_countries(
            countries=selected_countries,
            networks_folder=networks_folder,
            beta_values=beta_values,
            n_simulations=n_simulations,
            output_folder=output_folder,
            max_workers=max_workers
        )

        print("\n" + "=" * 70)
        print("Analysis completed!")
        print("=" * 70)

        if all_results:
            print("\nSummary statistics:")
            print("-" * 70)
            print(f"{'Country':<10}{'Nodes':<12}{'Mean degree':<12}{'Threshold β':<12}")
            print("-" * 70)
            for country, results in all_results.items():
                beta_results = results['beta_results']
                means = [br['mean_infected_ratio'] for br in beta_results]
                betas = [br['beta'] for br in beta_results]
                threshold = None
                for i, val in enumerate(means):
                    if val > 0.01:
                        threshold = betas[i]
                        break
                net_info = results['network_info']
                threshold_str = f"{threshold:.4f}" if threshold is not None else "N/A"
                print(f"{country:<10}{net_info['N']:<12,}{net_info['avg_degree']:<12.2f}{threshold_str:<12}")
        print(f"\nAll results saved to: {output_folder}")

    except Exception as e:
        print(f"\nRun failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()