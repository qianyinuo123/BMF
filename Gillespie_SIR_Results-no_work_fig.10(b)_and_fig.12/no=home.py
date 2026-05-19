# ============================================================
# Gillespie 年龄结构 SIR 模型（多进程并行版 - 内存优化）
# ============================================================

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings
import concurrent.futures as cf
import multiprocessing
import time
import random

warnings.filterwarnings('ignore')


# ============================================================
# 模块级函数：单个 beta 值模拟（优化内存版）
# ============================================================

def simulate_one_beta(model, G, beta, n_simulations, initial_infected_fraction):
    """
    对单个 beta 值执行 n_simulations 次独立模拟，返回统计结果（不存储完整轨迹）。
    """
    # 为每个子进程设置不同的随机种子
    seed = (os.getpid() * int(time.time())) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

    # 存储统计量（不存储完整模拟历史）
    infected_ratios = []
    R_all_values = []

    # 存储每个年龄组的最终感染比例（所有模拟）
    age_infections_all = {age: [] for age in range(16)}
    # 存储四大年龄组的最终感染比例（所有模拟）
    four_group_values = {
        "0-19": [],
        "20-39": [],
        "40-59": [],
        "60+": []
    }

    # 执行 n_simulations 次模拟
    for _ in range(n_simulations):
        sim_result = model.gillespie_sir_simulation(
            G, beta, initial_infected_fraction
        )
        infected_ratios.append(sim_result['final_infected_ratio'])
        R_all_values.append(sim_result['R_all_weighted'])

        # 收集年龄组感染比例
        for age in range(16):
            if age in sim_result['final_age_stats']:
                age_infections_all[age].append(
                    sim_result['final_age_stats'][age]['R_final']
                )
            else:
                age_infections_all[age].append(0.0)

        # 收集四大年龄组感染比例
        for gname in four_group_values.keys():
            four_group_values[gname].append(
                sim_result['four_group_stats'][gname]['R_final']
            )

    # ================= 总体统计 =================
    mean_val = np.mean(infected_ratios)
    sd_val = np.std(infected_ratios, ddof=1)
    ci95 = 1.96 * sd_val / np.sqrt(len(infected_ratios))

    # ================= 16 个年龄组统计 =================
    age_infected_means = {}
    age_infected_sds = {}
    age_infected_ci95_low = {}
    age_infected_ci95_high = {}

    for age in range(16):
        values = age_infections_all[age]
        if len(values) > 0:
            mean_age = np.mean(values)
            sd_age = np.std(values, ddof=1)
            ci95_age = 1.96 * sd_age / np.sqrt(len(values))
            age_infected_means[age] = mean_age
            age_infected_sds[age] = sd_age
            age_infected_ci95_low[age] = mean_age - ci95_age
            age_infected_ci95_high[age] = mean_age + ci95_age
        else:
            age_infected_means[age] = 0.0
            age_infected_sds[age] = 0.0
            age_infected_ci95_low[age] = 0.0
            age_infected_ci95_high[age] = 0.0

    # ================= 4 大年龄组统计 =================
    group_names = ["0-19", "20-39", "40-59", "60+"]
    four_group_summary = {}
    for gname in group_names:
        values = four_group_values[gname]
        mean_g = np.mean(values)
        sd_g = np.std(values, ddof=1)
        ci95_g = 1.96 * sd_g / np.sqrt(len(values))
        four_group_summary[gname] = {
            'mean': mean_g,
            'sd': sd_g,
            'ci95_low': mean_g - ci95_g,
            'ci95_high': mean_g + ci95_g
        }

    # 构建返回结果（不包含 simulations 列表，只包含统计量）
    beta_results = {
        'beta': beta,
        'R0': beta / model.gamma,
        'mean_infected_ratio': mean_val,
        'std_infected_ratio': sd_val,
        'sd_infected_ratio': sd_val,
        'ci95_low': mean_val - ci95,
        'ci95_high': mean_val + ci95,
        'min_infected_ratio': np.min(infected_ratios),
        'max_infected_ratio': np.max(infected_ratios),
        'median_infected_ratio': np.median(infected_ratios),
        'mean_R_all': np.mean(R_all_values),
        'age_infected_means': age_infected_means,
        'age_infected_sds': age_infected_sds,
        'age_infected_ci95_low': age_infected_ci95_low,
        'age_infected_ci95_high': age_infected_ci95_high,
        'four_group_summary': four_group_summary,
        # 为了兼容原有保存函数，添加一个空的 simulations 列表
        'simulations': []
    }
    return beta_results


# ============================================================
# 年龄结构 SIR 模型类（保持不变，除了绘图部分）
# ============================================================

class AgeStructuredSIRModel:
    """
    基于年龄结构的 Gillespie-SIR 模型
    """

    def __init__(self, gamma=0.1, max_time=200):
        self.gamma = gamma
        self.max_time = max_time

    # ------------------------------------------------------------
    # 加载网络
    # ------------------------------------------------------------
    def load_network(self, country, networks_folder):
        network_path = os.path.join(networks_folder, f"{country}.gexf")
        if not os.path.exists(network_path):
            print(f"❌ 网络文件不存在: {network_path}")
            return None
        try:
            G = nx.read_gexf(network_path)
            G = nx.convert_node_labels_to_integers(G)
            print(f"✅ 加载 {country} 网络: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
            return G
        except Exception as e:
            print(f"❌ 加载网络失败: {e}")
            return None

    # ------------------------------------------------------------
    # 提取年龄组
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Gillespie 连续时间 SIR 模拟（原版，未改动）
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 单国家分析（多进程并行版）
    # ------------------------------------------------------------
    def run_single_country_analysis(self, country, networks_folder, beta_values,
                                    n_simulations=50, initial_infected_fraction=0.001,
                                    max_workers=None):
        print(f"\n{'=' * 60}")
        print(f"开始分析: {country}")
        print(f"{'=' * 60}")

        G = self.load_network(country, networks_folder)
        if G is None:
            return None

        N = G.number_of_nodes()
        avg_degree = np.mean([d for n, d in G.degree()])
        print(f"网络信息: {N} 个节点, 平均度: {avg_degree:.2f}")

        age_groups = self.extract_age_groups(G)
        age_distribution = {}
        for age in range(16):
            count = sum(1 for ag in age_groups.values() if ag == age)
            if count > 0:
                age_distribution[age] = count
        print(f"年龄组分布: {age_distribution}")

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
        print(f"使用 {max_workers} 个进程并行处理 {len(beta_values)} 个 beta 值")

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
                               desc=f"并行模拟 - {country}"):
                beta_results_list.append(future.result())

        beta_results_list.sort(key=lambda x: x['beta'])
        results['beta_results'] = beta_results_list

        for br in beta_results_list:
            print(f"  Beta={br['beta']:.3f}: Mean={br['mean_infected_ratio']:.4f}, "
                  f"SD={br['std_infected_ratio']:.4f}, "
                  f"95%CI=[{br['ci95_low']:.4f}, {br['ci95_high']:.4f}]")

        return results

    # ------------------------------------------------------------
    # 多国家运行
    # ------------------------------------------------------------
    def run_multiple_countries(self, countries, networks_folder, beta_values,
                               n_simulations=50, output_folder="results",
                               max_workers=None):
        os.makedirs(output_folder, exist_ok=True)
        all_results = {}

        for country in countries:
            print(f"\n{'=' * 60}")
            print(f"处理国家: {country}")
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
                self.generate_country_report(country_results, output_folder)

        if len(all_results) > 1:
            self.generate_comparison_report(all_results, output_folder)

        return all_results

    # ------------------------------------------------------------
    # 保存原始 CSV（含 SD 和 95% CI）
    # ------------------------------------------------------------
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
                row[f'R_age{age + 1}'] = br['age_infected_means'].get(age, 0.0)
                row[f'R_age{age + 1}_sd'] = br['age_infected_sds'].get(age, 0.0)
                row[f'R_age{age + 1}_ci95_low'] = br['age_infected_ci95_low'].get(age, 0.0)
                row[f'R_age{age + 1}_ci95_high'] = br['age_infected_ci95_high'].get(age, 0.0)
            data_rows.append(row)

        df = pd.DataFrame(data_rows)
        csv_file = os.path.join(output_folder, f"Gillespie_SIR_{country}.csv")
        df.to_csv(csv_file, index=False)
        print(f"✅ 结果已保存: {csv_file}")
        return csv_file

    # ------------------------------------------------------------
    # 保存 4 大年龄组 CSV（直接使用预计算统计量）
    # ------------------------------------------------------------
    def save_four_group_results(self, country_results, output_folder):
        country = country_results['country']
        rows = []
        for br in country_results['beta_results']:
            row = {'Beta': br['beta']}
            group_names = ["0-19", "20-39", "40-59", "60+"]
            for gname in group_names:
                stats = br['four_group_summary'][gname]
                row[f'{gname}_mean'] = stats['mean']
                row[f'{gname}_sd'] = stats['sd']
                row[f'{gname}_ci95_low'] = stats['ci95_low']
                row[f'{gname}_ci95_high'] = stats['ci95_high']
            rows.append(row)
        df = pd.DataFrame(rows)
        save_path = os.path.join(output_folder, f"Gillespie_4AgeGroups_{country}.csv")
        df.to_csv(save_path, index=False)
        print(f"✅ 四年龄组结果保存: {save_path}")

    # ------------------------------------------------------------
    # 单国家报告（绘图，要求4个子图代替热图）
    # ------------------------------------------------------------
    def generate_country_report(self, country_results, output_folder):
        country = country_results['country']
        beta_results = country_results['beta_results']

        betas = [br['beta'] for br in beta_results]
        mean_infected = [br['mean_infected_ratio'] for br in beta_results]
        sd_infected = [br['std_infected_ratio'] for br in beta_results]
        ci95_low = [br['ci95_low'] for br in beta_results]
        ci95_high = [br['ci95_high'] for br in beta_results]

        plt.figure(figsize=(16, 12))

        # 子图1：最终感染规模
        plt.subplot(2, 2, 1)
        plt.plot(betas, mean_infected, linewidth=2, label='Mean Infection Ratio')
        plt.fill_between(betas, np.array(mean_infected) - np.array(sd_infected),
                         np.array(mean_infected) + np.array(sd_infected),
                         alpha=0.25, label='Mean ± SD')
        plt.fill_between(betas, ci95_low, ci95_high, alpha=0.2, label='95% CI')
        for i, val in enumerate(mean_infected):
            if val > 0.01:
                beta_threshold = betas[i]
                plt.axvline(beta_threshold, linestyle='--', linewidth=2,
                            label=f'Threshold≈{beta_threshold:.3f}')
                break
        plt.xlabel('Transmission Rate β', fontsize=12)
        plt.ylabel('Final Epidemic Size', fontsize=12)
        plt.title(f'{country} Gillespie-SIR', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()

        # 子图2：时间序列示例（使用第一次模拟的轨迹，但我们已经不存储轨迹了，所以需要重新运行一次？）
        # 为了不增加复杂度，这里可以跳过或使用一个占位图。我们改为显示一个说明。
        plt.subplot(2, 2, 2)
        plt.text(0.5, 0.5, 'Trajectory plot omitted\nfor memory efficiency',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
        # 原本需要轨迹数据，现在不再存储，所以简化显示。如需保留，可以单独运行一次示例模拟。

        # 子图3：四大年龄组（四个小图）
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        plt.subplot(2, 2, 3)
        plt.cla()
        gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=plt.gca().get_subplotspec(), wspace=0.3, hspace=0.3)
        group_names = ["0-19", "20-39", "40-59", "60+"]
        colors = ['blue', 'orange', 'green', 'red']
        for idx, gname in enumerate(group_names):
            ax = plt.subplot(gs[idx])
            means = [br['four_group_summary'][gname]['mean'] for br in beta_results]
            sds = [br['four_group_summary'][gname]['sd'] for br in beta_results]
            ax.plot(betas, means, linewidth=2, color=colors[idx])
            ax.fill_between(betas, np.array(means) - np.array(sds),
                            np.array(means) + np.array(sds), alpha=0.2, color=colors[idx])
            ax.set_title(gname, fontsize=10)
            ax.set_xlabel('β', fontsize=9)
            ax.set_ylabel('Recovered Fraction', fontsize=9)
            ax.grid(alpha=0.3)
        plt.suptitle('Four Age Groups (Mean ± SD)', fontsize=12)

        # 子图4：四大年龄组汇总曲线（保留原风格，可选）
        plt.subplot(2, 2, 4)
        for gname in group_names:
            means = [br['four_group_summary'][gname]['mean'] for br in beta_results]
            sds = [br['four_group_summary'][gname]['sd'] for br in beta_results]
            plt.plot(betas, means, linewidth=2, label=gname)
            plt.fill_between(betas, np.array(means) - np.array(sds),
                             np.array(means) + np.array(sds), alpha=0.15)
        plt.xlabel('Transmission Rate β', fontsize=12)
        plt.ylabel('Recovered Fraction', fontsize=12)
        plt.title('Four Large Age Groups (Comparison)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(output_folder, f"Gillespie_Report_{country}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 图已保存: {save_path}")

    # ------------------------------------------------------------
    # 多国家比较图
    # ------------------------------------------------------------
    def generate_comparison_report(self, all_results, output_folder):
        plt.figure(figsize=(14, 10))
        for country, results in all_results.items():
            betas = [br['beta'] for br in results['beta_results']]
            means = [br['mean_infected_ratio'] for br in results['beta_results']]
            plt.plot(betas, means, linewidth=2, label=country)
        plt.xlabel('Transmission Rate β', fontsize=12)
        plt.ylabel('Final Epidemic Size', fontsize=12)
        plt.title('Multi-country Gillespie-SIR Comparison', fontsize=15)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(output_folder, "MultiCountry_Comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 多国家比较图已保存")


# ============================================================
# main 函数
# ============================================================
def main():
    networks_folder = "gexf_networks-泊松-151-第二种组间连接——10000-没有工作"
    output_folder = "Gillespie_SIR_Results-没有工作"
    selected_countries = ["Qatar"]  # 可改为 ["Uganda", "Qatar", "Monaco", "Germany"]

    gamma = 1 / 3
    initial_infected_fraction = 0.001
    max_time = 200

    beta_min = 0.0
    beta_max = 0.4
    num_beta = 101  # 可以增加到401，但建议先用101测试
    beta_values = np.linspace(beta_min, beta_max, num_beta)

    n_simulations = 30
    max_workers = 30  # 降低并发数以避免内存压力（可根据实际内存调整）

    print("=" * 70)
    print("年龄结构 Gillespie-SIR 模型（多进程并行版 - 内存优化）")
    print("=" * 70)
    print(f"gamma = {gamma}")
    print(f"initial infected fraction = {initial_infected_fraction}")
    print(f"max_time = {max_time}")
    print(f"beta range = {beta_min} ~ {beta_max}")
    print(f"beta number = {num_beta}")
    print(f"independent realizations = {n_simulations}")
    print(f"countries = {selected_countries}")
    print(f"并行进程数 = {max_workers}")
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
        print("分析完成！")
        print("=" * 70)

        if all_results:
            print("\n总结统计")
            print("-" * 70)
            print(f"{'国家':<10}{'节点数':<12}{'平均度':<12}{'阈值β':<12}")
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
        print(f"\n✅ 所有结果已保存到: {output_folder}")

    except Exception as e:
        print(f"\n❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()