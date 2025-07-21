import os
import tempfile
from datasets import load_dataset
from typing import List
import dspy
from dotenv import load_dotenv
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate

# 加载 OpenAI API 密钥
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 配置语言模型
dspy.configure(lm=dspy.LM('gpt-4o-mini'))

# 定义情感标签
EMOTION_LABELS = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval',
    5: 'caring', 6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment',
    10: 'disapproval', 11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear',
    15: 'gratitude', 16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness',
    20: 'optimism', 21: 'pride', 22: 'realization', 23: 'relief', 24: 'remorse',
    25: 'sadness', 26: 'surprise', 27: 'neutral'
}


# 数据加载与预处理
def load_go_emotion_dataset() -> dict:
    """
    加载 GoEmotions 数据集，选择 simplified 配置。
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["HF_DATASETS_CACHE"] = temp_dir
        return load_dataset("google-research-datasets/go_emotions", "simplified", trust_remote_code=True)


def preprocess_labels(labels):
    """
    将整数列表转换为情感标签字符串列表。
    """
    if isinstance(labels, int):
        return [EMOTION_LABELS[labels]]  # 将单个整数标签映射为字符串
    elif isinstance(labels, list):
        return [EMOTION_LABELS[label] for label in labels if label in EMOTION_LABELS]  # 过滤无效标签
    else:
        raise ValueError(f"Unexpected label format: {labels}")


def prepare_go_emotions_dataset(data_split, start: int, end: int) -> List[dspy.Example]:
    """
    将指定范围的 GoEmotions 数据集切片转化为 DSPy 的 Example 格式。
    """
    examples = []
    for row in data_split.select(range(start, end)):
        text = row["text"]
        labels = preprocess_labels(row["labels"])  # 转换为字符串标签列表
        examples.append(
            dspy.Example(
                text=text,
                expected_labels=labels
            ).with_inputs("text")
        )
    return examples


# 加载数据集
dataset = load_go_emotion_dataset()

# 准备训练集和测试集
train_set = prepare_go_emotions_dataset(dataset["train"], 0, 50)
test_set = prepare_go_emotions_dataset(dataset["test"], 0, 50)

# 检查数据格式
print("检查数据集样本格式：")
for example in train_set[:5]:
    print(example)


# 签名与分类模块定义
class EmotionClassification(dspy.Signature):
    """
    情感分类任务的签名，输入是文本，输出是 1 到 5 个情感标签的字符串列表。
    """
    text: str = dspy.InputField(desc="Input text to classify emotions from")
    extracted_emotions: List[str] = dspy.OutputField(
        desc=(
            f"Analyze the input text and extract the most relevant 1 to 5 emotions. "
            f"The emotions must be chosen from the following list: {list(EMOTION_LABELS.values())}. "
            f"Your output should reflect the emotional nuances of the text."
        ),
        min_items=1,
        max_items=5
    )


# 使用 Chain of Thought 分类器
student_predictor = dspy.ChainOfThoughtWithHint(EmotionClassification)


# 定义情感解析函数
def parse_extracted_emotions(raw_output) -> List[str]:
    """
    将模型输出的整数列表情感标签转换为对应的字符串标签列表。
    """
    if isinstance(raw_output, list):
        return [EMOTION_LABELS[label] for label in raw_output if label in EMOTION_LABELS]
    elif isinstance(raw_output, int):
        return [EMOTION_LABELS[raw_output]]
    else:
        raise ValueError(f"Unexpected format for extracted_emotions: {raw_output}")


# 定义 Jaccard 相似度指标
def jaccard_similarity_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    计算 Jaccard 相似度，确保预测值转换为字符串标签。
    """
    predicted_emotions = set(parse_extracted_emotions(prediction.extracted_emotions))
    actual_emotions = set(example.expected_labels)

    # 打印调试信息
    print(f"Predicted: {predicted_emotions}, Actual: {actual_emotions}")

    if not predicted_emotions and not actual_emotions:
        return 1.0  # 如果预测和实际都为空，完全匹配
    intersection = predicted_emotions.intersection(actual_emotions)
    union = predicted_emotions.union(actual_emotions)
    return len(intersection) / len(union) if union else 0.0


# Few-shot 示例
few_shot_examples = [
    dspy.Example(text="I feel so sad and lonely.", expected_labels=["sadness"]),
    dspy.Example(text="This is amazing! I love it.", expected_labels=["joy", "excitement"]),
    dspy.Example(text="I’m really grateful for your help.", expected_labels=["gratitude"]),
    dspy.Example(text="What a confusing situation!", expected_labels=["confusion"]),
    dspy.Example(text="This is disgusting and unacceptable!", expected_labels=["disgust", "anger"]),
]

# 使用 MIPROv2 优化器进行微调
optimizer = MIPROv2(
    metric=jaccard_similarity_metric,
    auto="medium"  # 自动优化级别，可选 "light", "medium", "heavy"
)

# 编译和微调模型
optimized_classifier = optimizer.compile(
    student=student_predictor,
    trainset=train_set,
    valset=test_set,  # 可选验证集
    max_bootstrapped_demos=5,  # 设置少样本提示的最大数量
    requires_permission_to_run=False
)

# 模型推理与测试
print("\n测试单条样本的推理结果：")
sample_input = {"text": "I feel really happy and excited about this!"}
prediction = optimized_classifier.forward(**sample_input)
print(f"Prediction: {prediction}")

# 在测试集上验证模型性能
print("\n在测试集上的评估结果：")
evaluate_correctness = Evaluate(
    devset=test_set,
    metric=jaccard_similarity_metric,
    num_threads=8,
    display_progress=True,
    display_table=True
)
evaluate_correctness(optimized_classifier, devset=test_set)