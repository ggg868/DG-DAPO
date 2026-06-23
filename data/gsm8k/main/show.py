import pandas as pd

# 请将这里的路径替换为你实际的文件名
file_path = "test-00000-of-00001.parquet" 

# 读取 parquet 文件
df = pd.read_parquet(file_path)

print("=== Parquet 文件检查报告 ===")
print(f"总行数: {len(df)}")
print(f"字典键名 (Columns): {df.columns.tolist()}")

print("\n=== 前 5 行内容 ===")
# 强制 pandas 取消列和列宽截断，确保长文本完整显示
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# 使用 .head(5) 只输出前五行
print(df.head(5))