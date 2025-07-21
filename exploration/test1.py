from sklearn.metrics import accuracy_score

# 创建标签到数字标号的映射
label_to_id = {
    'admiration': 0,
    'amusement': 1,
    'anger': 2,
    'annoyance': 3,
    'approval': 4,
    'caring': 5,
    'confusion': 6,
    'curiosity': 7,
    'desire': 8,
    'disappointment': 9,
    'disapproval': 10,
    'disgust': 11,
    'embarrassment': 12,
    'excitement': 13,
    'fear': 14,
    'gratitude': 15,
    'grief': 16,
    'joy': 17,
    'love': 18,
    'nervousness': 19,
    'optimism': 20,
    'pride': 21,
    'realization': 22,
    'relief': 23,
    'remorse': 24,
    'sadness': 25,
    'surprise': 26,
    'neutral': 27
}


def compute_metrics(labels, predictions):
    # 打印调试信息，确保 labels 和 predictions 格式正确
    print("Labels:", labels)
    print("Predictions:", predictions)

    # 将预测标签转换为数字标号
    converted_predictions = []
    for pred in predictions:
        if isinstance(pred, str) and pred in label_to_id:
            converted_predictions.append(label_to_id[pred])
        else:
            # 如果预测的标签不在映射中，设置为一个默认值（如 -1）
            converted_predictions.append(-1)

    # 确保 labels 和 predictions 长度一致
    if len(labels) != len(converted_predictions):
        print(f"Labels 和 Predictions 长度不一致: {len(labels)} vs {len(converted_predictions)}")
        return 0

    # 计算准确率
    accuracy = accuracy_score(labels, converted_predictions)
    return accuracy


def evaluate(model):
    # 模拟从数据集中获取 labels 和 predictions 的部分
    # 假设模型预测返回的是一个包含标签名称的字符串列表
    labels = [0, 1, 2, 3, 4]  # 示例 labels（实际数据请替换）
    predictions = ['admiration', 'amusement', 'anger', 'annoyance', 'caring']  # 示例 predictions

    # 计算 metrics
    accuracy = compute_metrics(labels, predictions)
    print(f"Accuracy: {accuracy}")


# 调用 evaluate 函数
evaluate(None)  # 这里只是一个示例，传入模型参数
