# from openpyxl import load_workbook
#
# # 打开 Excel 文件
# wb = load_workbook("Japan-2023.xlsx")
#
# # 获取表单（sheet）
# ws = wb.active  # 当前活动表
# # 或按名字获取
# # ws = wb["Sheet1"]
#
# # 读取单元格
# value = ws["A1"].value
# print(value)
#
# # 遍历行列
# for row in ws.iter_rows(min_row=1, max_row=5, values_only=True):
#     print(row)
#
# print(111)
# col = ws['Age']
#
# # 输出这一列的内容
# print(col)


import pandas as pd

# 只读取第5列（E列）
df = pd.read_excel("Japan-2023.xlsx", usecols=[4])  # 索引从0开始，所以4表示E列

print(df.head(21))   # 查看前20行
