import pandas as pd

# 直接读取你下载好的本地文件
df = pd.read_parquet("train-00000-of-00001.parquet")
df.to_json("squad_train.json", orient="records", force_ascii=False)
print("转换成功！")

# 读取你下载好的验证集 parquet 文件
df_val = pd.read_parquet("validation-00000-of-00001.parquet")

# 导出为 JSON 文件
df_val.to_json("squad_validation.json", orient="records", force_ascii=False)

print("验证集转换成功！")