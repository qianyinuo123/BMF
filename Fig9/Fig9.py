import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 读取数据（假设csv文件在当前目录）
df = pd.read_csv('network_structure_comparison_interventions.csv')

# 确保场景顺序（标签改为首字母大写）
scenario_order = ['baseline', 'no_school', 'no_work']
scenario_labels = {'baseline': 'Baseline', 'no_school': 'No_school', 'no_work': 'No_work'}
df['scenario'] = pd.Categorical(df['scenario'], categories=scenario_order, ordered=True)

# 国家顺序
countries = ['Germany', 'Monaco', 'Qatar', 'Uganda']
df['country'] = pd.Categorical(df['country'], categories=countries, ordered=True)

# 颜色映射
colors = {'baseline': '#1f77b4', 'no_school': '#ff7f0e', 'no_work': '#2ca02c'}

# ================= 图1：分组柱状图（四个指标） =================
metrics = [
    ('cross_edge_ratio', 'Cross-age edge ratio', 0, 1),
    ('avg_path_length', 'Average shortest path length', None, None),
    ('mean_degree', 'Mean degree', None, None),
    ('largest_comp_frac', 'Largest component fraction', 0.9, 1.0)
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, (metric, label, ymin, ymax) in enumerate(metrics):
    ax = axes[idx]
    x = np.arange(len(countries))
    width = 0.25

    for i, scenario in enumerate(scenario_order):
        values = []
        for c in countries:
            val = df[(df['country'] == c) & (df['scenario'] == scenario)][metric].values[0]
            values.append(val)
        bars = ax.bar(x + i * width, values, width, label=scenario_labels[scenario], color=colors[scenario])
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(countries)
    ax.set_ylabel(label)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)
    ax.set_title('')   # 移除标题
    ax.legend()

plt.tight_layout()
plt.savefig('network_metrics_grouped_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('network_metrics_grouped_bars.pdf', bbox_inches='tight')
plt.show()

# ================= 图2：跨年龄边比例相对变化柱状图 =================
# 计算相对变化百分比
change_data = []
for c in countries:
    base = df[(df['country'] == c) & (df['scenario'] == 'baseline')]['cross_edge_ratio'].values[0]
    school = df[(df['country'] == c) & (df['scenario'] == 'no_school')]['cross_edge_ratio'].values[0]
    work = df[(df['country'] == c) & (df['scenario'] == 'no_work')]['cross_edge_ratio'].values[0]
    change_data.append({
        'country': c,
        'school_change_pct': (school - base) / base * 100,
        'work_change_pct': (work - base) / base * 100
    })
change_df = pd.DataFrame(change_data)

x = np.arange(len(countries))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width / 2, change_df['school_change_pct'], width, label='No_school', color='#ff7f0e')
ax.bar(x + width / 2, change_df['work_change_pct'], width, label='No_work', color='#2ca02c')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(countries)
ax.set_ylabel('Relative change in cross-age edge ratio (%)')
ax.set_title('Intervention effect on cross-age connectivity')
ax.legend()

# 添加数值标签
for i, (school, work) in enumerate(zip(change_df['school_change_pct'], change_df['work_change_pct'])):
    ax.text(i - width / 2, school + 0, f'{school:.1f}%', ha='center',
            va='bottom' if school > 0 else 'top', fontsize=9)
    ax.text(i + width / 2, work + (1 if work > 0 else -0.2), f'{work:.1f}%', ha='center',
            va='bottom' if work > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig('cross_edge_relative_change.png', dpi=300, bbox_inches='tight')
plt.savefig('cross_edge_relative_change.pdf', bbox_inches='tight')
plt.show()

# 可选：生成一个带误差的表格（直接打印）
print("\n关键指标汇总表（可复制到论文中）")
print(df[['country', 'scenario', 'cross_edge_ratio', 'avg_path_length', 'mean_degree',
          'largest_comp_frac']].round(4))