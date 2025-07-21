# 1. 导入必要库
from dotenv import load_dotenv
import os
import random
import pandas as pd
import dspy
from dspy.evaluate import Evaluate
from sklearn.metrics import f1_score, accuracy_score
from typing import Literal

# 2. 加载 .env 文件和 OpenAI API 密钥
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 配置 GPT-3.5-turbo 模型
turbo = dspy.LM(model="openai/gpt-3.5-turbo", max_tokens=250, api_key=api_key)
dspy.configure(lm=turbo)

# 3. 加载 .parquet 文件
trainset = pd.read_parquet(r"C:\Users\xinjiang\Downloads\train-00000-of-00001.parquet")
devset = pd.read_parquet(r"C:\Users\xinjiang\Downloads\validation-00000-of-00001.parquet")
testset = pd.read_parquet(r"C:\Users\xinjiang\Downloads\test-00000-of-00001.parquet")

# 4. 随机抽取10个样本用于快速评估
sample_trainset = random.sample(trainset.to_dict(orient="records"), 10)
sample_devset = random.sample(devset.to_dict(orient="records"), 10)
sample_testset = random.sample(testset.to_dict(orient="records"), 10)


# 5. 将数据集包装成 DSPy 的 Example 对象并指定输入字段
def wrap_example(dataset):
    examples = []
    for entry in dataset:
        labels = entry.get('labels', [])
        if isinstance(labels, str):  # 如果是字符串格式，尝试转换为列表
            try:
                labels = eval(labels)  # 转换字符串为列表
            except Exception as e:
                print(f"Error evaluating labels for {entry.get('id', 'unknown')}: {e}")
                labels = []  # 如果出错，则返回空列表
        if isinstance(labels, int):  # 确保标签是一个列表
            labels = [labels]
        example = dspy.Example(text=entry['text'], labels=labels)
        example = example.with_inputs('text')  # 设置输入字段
        examples.append(example)
    return examples


# 包装随机抽取的样本集
trainset = wrap_example(sample_trainset)
devset = wrap_example(sample_devset)
testset = wrap_example(sample_testset)

# 6. 定义情感标签字典（0-27）
class_labels = {
    '0': 'admiration', '1': 'amusement', '2': 'anger', '3': 'annoyance', '4': 'approval',
    '5': 'caring', '6': 'confusion', '7': 'curiosity', '8': 'desire', '9': 'disappointment',
    '10': 'disapproval', '11': 'disgust', '12': 'embarrassment', '13': 'excitement', '14': 'fear',
    '15': 'gratitude', '16': 'grief', '17': 'joy', '18': 'love', '19': 'nervousness',
    '20': 'optimism', '21': 'pride', '22': 'realization', '23': 'relief', '24': 'remorse',
    '25': 'sadness', '26': 'surprise', '27': 'neutral'
}


# 7. 定义文本分类模型
class TextClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict("text -> labels")

    def forward(self, text):
        predictions = self.prog(text=text)

        # 暂时输出预测的格式，检查其结构
        print(f"Predictions: {predictions}")

        # 处理预测结果：提取标签并确保其在 0-27 标签范围内
        if isinstance(predictions, str):  # 如果是字符串，假设是单一标签
            predictions = [predictions]

        # 如果预测是以逗号分隔的多个标签，进行分割
        if isinstance(predictions, list) and isinstance(predictions[0], str):
            predictions = [label.strip() for label in predictions[0].split(',')]

        # 确保预测标签在 0-27 范围内，并转换为数字
        valid_labels = [label for label in predictions if label in class_labels.values()]
        label_ids = [key for key, value in class_labels.items() if value in valid_labels]

        return label_ids  # 返回标签 ID（0-27 之间）


# 8. 统一标签和预测的格式
def format_labels(labels):
    """将标签或预测值转换为扁平化的整数列表格式"""
    if isinstance(labels, list):
        return [int(label) for label in labels]  # 确保是整数列表
    elif isinstance(labels, str):
        try:
            return [int(x) for x in eval(labels)]
        except Exception:
            return []
    elif isinstance(labels, int):
        return [labels]
    return []


# 9. 定义评估指标函数
def compute_metrics(predictions, labels):
    # 转换 labels 和 predictions 为一致的格式
    formatted_labels = format_labels(labels)
    formatted_predictions = format_labels(predictions)

    # 如果格式为空，跳过评估
    if not formatted_labels or not formatted_predictions:
        print("Empty labels or predictions encountered.")
        return {"accuracy": 0, "f1": 0}

    # 计算准确率和 F1 分数
    accuracy = accuracy_score(formatted_labels, formatted_predictions)
    f1 = f1_score(formatted_labels, formatted_predictions, average='weighted')  # 使用加权平均
    return {"accuracy": accuracy, "f1": f1}


# 10. 使用 Evaluate 模块对模型进行评估
evaluate = Evaluate(
    devset=devset,
    metric=compute_metrics,  # 设置评估指标
    num_threads=4,
    display_progress=True,
    display_table=True,
    provide_traceback=True  # 启用详细错误堆栈
)

# 定义模型
baseline_model = TextClassifier()

# 在验证集上评估模型
evaluate(baseline_model)

# 11. 在测试集上评估最终模型
evaluate = Evaluate(
    devset=testset,
    metric=compute_metrics,  # 设置评估指标
    num_threads=4,
    display_progress=True,
    display_table=True,
    provide_traceback=True  # 启用详细错误堆栈
)
evaluate(baseline_model)  # 在测试集上评估模型
