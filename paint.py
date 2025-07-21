# ==============================================================================
# 项目：GoEmotion 数据集探索性数据分析 (EDA) - 最终修正版 v6
# 目的：通过强制后端和调整顺序，终结中文显示问题
# ==============================================================================

# 【关键修正 1】在导入pyplot之前，强制指定后端为'Agg'
import matplotlib

matplotlib.use('Agg')  # 必须在导入pyplot之前设置

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from matplotlib import font_manager


def set_global_chinese_font():
    """
    设置Matplotlib的全局字体，以支持中文显示。
    """
    try:
        font_path = 'C:/Windows/Fonts/simsun.ttc'
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 成功将全局字体设置为: {font_prop.get_name()} (来自 {font_path})")
    except Exception as e:
        print(f"❌ 设置全局中文字体失败: {e}")


def main():
    """
    主函数，执行数据加载、分析和可视化的完整流程。
    """

    # --- 步骤 1: 加载数据集 ---
    print("🚀 开始加载GoEmotion数据集...")
    try:
        dataset_dict = load_dataset("google-research-datasets/go_emotions", trust_remote_code=True)
        dataset = dataset_dict['train']
        print("✅ 数据集加载成功！")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    # --- 步骤 2: 转换为Pandas DataFrame ---
    print("\n🔄 正在将数据集转换为Pandas DataFrame...")
    df = dataset.to_pandas()
    print("✅ 转换完成！")

    # --- 步骤 3: 生成图2.1 ---
    print("\n📊 正在生成图2.1: 情绪标签分布图...")
    all_labels_indices = [label for sublist in df['labels'] for label in sublist]
    label_names = dataset.features['labels'].feature.names
    label_counts = pd.Series(all_labels_indices).value_counts()
    label_counts.index = [label_names[i] for i in label_counts.index]

    # 【关键修正 2】调整设置顺序：先设置样式，再设置字体
    plt.style.use('seaborn-v0_8-whitegrid')
    set_global_chinese_font()  # 在应用样式后，再次确保中文字体设置生效

    plt.figure(figsize=(12, 10))
    sns.barplot(x=label_counts.values, y=label_counts.index, orient='h', palette='viridis')

    # plt.title('图2.1: GoEmotion数据集中各情绪标签的频率分布', fontsize=16, pad=20)
    plt.xlabel('频率 (Frequency)', fontsize=12)
    plt.ylabel('情绪类别 (Emotion Category)', fontsize=12)

    plt.tight_layout()
    output_filename_1 = 'emotion_distribution_final.png'
    plt.savefig(output_filename_1, dpi=300)
    print(f"✅ 图2.1已成功保存为 '{output_filename_1}'")
    # 由于使用了'Agg'后端，plt.show()不会弹出窗口，这是正常的。
    # plt.show()

    # --- 步骤 4: 生成图2.2 ---
    print("\n📊 正在生成图2.2: 文本长度分布图...")
    df['text_length'] = df['text'].str.len()

    # 再次确保字体设置
    plt.style.use('seaborn-v0_8-whitegrid')
    set_global_chinese_font()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['text_length'], bins=50, kde=True, color='skyblue', edgecolor='black')

    # plt.title('图2.2: GoEmotion数据集中文本长度的分布情况', fontsize=16, pad=20)
    plt.xlabel('文本长度 (Number of Characters)', fontsize=12)
    plt.ylabel('样本数量 (Number of Samples)', fontsize=12)

    plt.xlim(0, max(800, df['text_length'].quantile(0.99)))
    output_filename_2 = 'text_length_distribution_final.png'
    plt.savefig(output_filename_2, dpi=300)
    print(f"✅ 图2.2已成功保存为 '{output_filename_2}'")
    # plt.show()


if __name__ == '__main__':
    main()