
import pandas as pd

# 读取xlsx文件
df = pd.read_excel('Angola-2023.xlsx', header=None)  # header=None 确保不将第一行作为列名

# 读取第4列第23行 (索引从0开始，所以是第3列第22行)
value = df.iloc[22, 3]  # iloc[行索引, 列索引]
print(value)