import dspy

# 加载 simplified 配置的数据集
dataset = Dataset.from_huggingface("goemotions", config="simplified")

# 检查数据
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

print(train_data[:5])  # 查看前5条数据
