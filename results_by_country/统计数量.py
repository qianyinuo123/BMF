from pathlib import Path

folder = Path("")  # 改成实际路径

files = list(folder.glob("Poison_beta_*.csv"))

print(f"国家数量: {len(files)}")
