import pandas as pd

# 请将这里的路径替换为你实际的文件名，例如 "train-00000-of-00001.parquet"
file_path = "train-00000-of-00001.parquet" 

# 读取 parquet 文件
df = pd.read_parquet(file_path)

print("=== Parquet 文件检查报告 ===")
print(f"总行数: {len(df)}")
print(f"字典键名 (Columns): {df.columns.tolist()}")

print("\n=== 全部内容 ===")
# 强制 pandas 取消截断，输出所有行、所有列以及完整的文本内容
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

print(df)