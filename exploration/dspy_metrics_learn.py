# dspy指标学习
import functools
#
import os
import random

import dspy
from dotenv import load_dotenv
from dsp.utils import deduplicate
from dspy import Predict, ensure_signature, Module
from dspy.predict.parameter import Parameter
from nltk.metrics import edit_distance  # 导入编辑距离
from sklearn.model_selection import train_test_split

# 加载 .env 文件
load_dotenv()

# 获取 OpenAI API 密钥
api_key = os.getenv("OPENAI_API_KEY")

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm, experimental=True)


def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()


def validate_context_and_answer(example, pred, trace=None):
    # check the gold label and the predicted answer are the same
    answer_match = example.answer.lower() == pred.answer.lower()

    # check the predicted answer comes from one of the retrieved contexts
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    if trace is None:  # if we're doing evaluation or optimization
        return (answer_match + context_match) / 2.0
    else:  # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step
        return answer_match and context_match


def validate_context_and_answer_and_hops(example, pred, trace=None):
    """
    Validates the context and answer, considering the multi-hop reasoning process.

    :param example: The gold label example containing the correct answer and context.
    :param pred: The predicted result, including the answer and context.
    :param trace: Optional parameter to handle bootstrapping or training processes.
    :return: A score indicating how well the predicted answer and context match the gold label.
    """
    # Step 1: Check if the predicted answer matches the true answer (ignoring case)
    answer_match = example.answer.lower() == pred.answer.lower()

    # Step 2: Check if the predicted answer is contained in any of the retrieved contexts (any hop)
    context_match = any((pred.answer.lower() in c.lower()) for c in pred.context)

    # Step 3: Check if the hops are valid. For multi-hop reasoning, we need to check each hop's context.
    hops_valid = True
    for hop_num, hop_context in enumerate(pred.context):
        # Example: Ensure that each hop context is relevant to the question and the reasoning process
        if not validate_hop_context(example, hop_context, hop_num):
            hops_valid = False
            break

    # Step 4: Calculate final validation score based on the answer match, context match, and hop validation
    if trace is None:  # if we're doing evaluation or optimization
        # Average score of answer match, context match, and hops validity
        return (answer_match + context_match + hops_valid) / 3.0
    else:  # if we're doing bootstrapping (self-generating good demonstrations)
        return answer_match and context_match and hops_valid


def validate_hop_context(example, hop_context, hop_num):
    """
    Validates the context for each hop in the multi-hop reasoning process.
    This is a placeholder function that can be extended depending on the specific validation logic for each hop.

    :param example: The gold label example.
    :param hop_context: The context for the current hop.
    :param hop_num: The current hop number.
    :return: True if the hop context is valid, False otherwise.
    """
    # Example logic to validate hop context (this can be customized)
    # For now, we simply check if the hop context contains some keyword from the question
    # which may indicate relevance. You can add more sophisticated validation logic.
    if hop_num == 0:  # First hop context validation
        return any(keyword in hop_context.lower() for keyword in example.query.lower().split())
    else:
        return True  # For other hops, you may apply different validation rules (e.g., consistency with previous hops)


# # 测试
# from dspy.datasets import DataLoader
# import dspy
# from nltk.metrics import edit_distance  # 导入编辑距离
# import re  # 导入正则表达式模块
#
# # 加载数据集
# dl = DataLoader()
#
# # 从 Hugging Face 下载数据集
# code_alpaca = dl.from_huggingface("ruslanmv/ai-medical-chatbot")
#
# # 获取训练集
# devset = code_alpaca['train']
#
# # 创建 ChainOfThought 模型，输入格式： 'question -> answer'
# classify = dspy.ChainOfThought('question -> answer', n=5)
#
# # 初始化评分列表
# scores = []
#
# # 遍历数据集，处理前 1 个样本
# for i, x in enumerate(devset):
#     if i >= 1:  # 如果已经处理了 1 个样本，停止循环
#         break
#
#     # 提取 Description 和 Patient 字段
#     description = x['Description']
#     patient_input = x['Patient']
#
#     # 将 Description 和 Patient 作为输入来生成医生的回答
#     inputs_data = {'question': f"Patient: {patient_input} \nDescription: {description}", 'answer': ''}
#
#     # 使用模型生成医生的回答
#     pred = classify(**inputs_data)
#
#     # 如果 pred 是一个对象，提取实际的文本内容
#     # 假设 pred 是一个字典，且生成的回答存储在 'answer' 字段
#     pred_answer = pred['answer'] if isinstance(pred, dict) else str(pred)  # 根据实际情况提取回答
#
#     # 获取真实医生的回答
#     real_doctor_answer = x['Doctor']
#
#     # 确保这两个值是字符串类型
#     real_doctor_answer_str = str(real_doctor_answer)  # 转为字符串
#     pred_answer_str = str(pred_answer)  # 转为字符串
#
#     # 使用正则表达式从 pred_answer_str 中提取 'answer=' 后面的内容
#     # 正则表达式提取 answer 后面的纯文本内容
#     match = re.search(r"answer=['\"](.*?)['\"]", pred_answer_str)
#     if match:
#         pred_answer_str = match.group(1)  # 提取答案部分
#
#     # 输出真实医生的回答和模型生成的回答，查看格式
#     print("Real Doctor Answer:")
#     print(real_doctor_answer_str)
#     print("\nPredicted Answer:")
#     print(pred_answer_str)
#     print("\n")
#
#     # 计算编辑距离
#     score = edit_distance(real_doctor_answer_str, pred_answer_str)
#
#     # 将评分添加到列表
#     scores.append(score)
#
# # 打印评分结果
# print(scores)


# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")


gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=1000, model_type='chat')


def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"

    with dspy.context(lm=gpt4T):
        correct = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct)
        engaging = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging)

    correct, engaging = [m.assessment_answer.lower() == 'yes' for m in [correct, engaging]]
    score = (correct + engaging) if correct and (len(tweet) <= 280) else 0

    if trace is not None: return score >= 2
    return score / 2.0


from pydantic import BaseModel, Field
import dspy


class Retrieve(Parameter):
    def __init__(self, k=3):
        self.stage = random.randbytes(8).hex()
        self.k = k


# Example Usage

# Define a retrieval model server to send retrieval requests to
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# Configure retrieval server internally
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

# Define Retrieve Module
retriever = dspy.Retrieve(k=3)


# Python function for validating distinct queries
def validate_query_distinction_local(previous_queries, query):
    """Check if query is distinct from previous queries"""
    if previous_queries == []:
        return True
    if dspy.evaluate.answer_exact_match_str(query, previous_queries, frac=0.8):
        return False
    return True


class Input(BaseModel):
    context: str = Field(description="The context for the question")
    query: str = Field(description="The question to be answered")


class Output(BaseModel):
    answer: str = Field(description="The answer for the question")
    confidence: float = Field(ge=0, le=1, description="The confidence score for the answer")


class QASignature(dspy.Signature):
    """Answer the question based on the context and query provided, and on the scale of 10 tell how confident you are
    about the answer."""
    input: Input = dspy.InputField()
    output: Output = dspy.OutputField()


def all_queries_distinct(queries):
    return len(queries) == len(set(queries))


# Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class SimplifiedBaleenAssertions(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ReAct('question -> answer') for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ReAct(BasicQA)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        for hop in range(self.max_hops):
            # 直接获取 query 的结果，而不是尝试从 Prediction 中提取 query
            prediction = self.generate_query[hop](context=context, question=question)
            query = prediction.answer  # 或者根据返回的结果来选择适当的字段

            # 处理验证条件
            dspy.Suggest(
                len(query) <= 500,
                "Query should be short and less than 500 characters",
                target_module=self.generate_query
            )

            dspy.Suggest(
                validate_query_distinction_local(prev_queries, query),
                "Query should be distinct from: "
                + "; ".join(f"{i + 1}) {q}" for i, q in enumerate(prev_queries)),
                target_module=self.generate_query
            )

            prev_queries.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        if all_queries_distinct(prev_queries):
            self.passed_suggestions += 1

        # 生成最终的答案
        pred = self.generate_answer(context=context, question=question)
        pred = dspy.Prediction(context=context, answer=pred.answer)

        return pred


# 激活 assertions 的 Baleen 模型
baleen_with_assertions = SimplifiedBaleenAssertions().activate_assertions()

# # 传入问题文本
# question = ("Who acted in the short film The Shore and is also the youngest actress ever to play Ophelia in a Royal "
#             "Shakespeare Company production of \"Hamlet\"  ?")
#
# # 调用 forward 方法进行推理（确保传递了 question 参数）
# prediction = baleen_with_assertions.forward(question=question)
#
# # 打印预测结果
# print(f"Prediction Context: {prediction.context}")
# print(f"Answer: {prediction.answer}")
#
# # 调用 inspect_history 打印历史记录
# lm.inspect_history(n=3)


import random


def max_bootstrapped_demos(examples, max_demos):
    """
    Generates a maximum of `max_demos` bootstrapped examples from a given list of examples.

    :param examples: List of examples to bootstrap from (can be model predictions or training data).
    :param max_demos: Maximum number of bootstrapped examples to generate.
    :return: A list of bootstrapped examples.
    """
    # Check if the number of examples is less than or equal to the max_demos
    if len(examples) <= max_demos:
        return examples  # If fewer examples than max, return all of them

    # Randomly sample `max_demos` examples from the list
    bootstrapped_examples = random.sample(examples, max_demos)

    return bootstrapped_examples


# teleprompter = dspy.BootstrapFewShotWithRandomSearch(
#     metric=validate_context_and_answer_and_hops,
#     max_bootstrapped_demos=max_bootstrapped_demos,
#     num_candidate_programs=6,
# )

from dspy.datasets import DataLoader
from sklearn.model_selection import train_test_split

# 加载数据集
dl = DataLoader()
code_alpaca = dl.from_huggingface("ruslanmv/ai-medical-chatbot")

# 仅取出前50个样本
limited_dataset = code_alpaca['train'][:50]

# 划分数据集，将50个样本划分为训练集和验证集
trainset, devset = train_test_split(limited_dataset, test_size=0.2, random_state=42)


# 创建 Example 类，只使用 Patient 作为 question
class Example:
    def __init__(self, data):
        self.data = data

    def inputs(self):
        # 只返回 Patient 作为 question
        return {'question': self.data['Patient']}


# 转换原始字典为对象
trainset = [Example(example) for example in trainset]
devset = [Example(example) for example in devset]

# # Compilation with Assertions
# compiled_with_assertions_baleen = teleprompter.compile(
#     student=SimplifiedBaleenAssertions(),
#     teacher=baleen_with_assertions,
#     trainset=trainset,
#     valset=devset
# )
#
# # Compilation + Inference with Assertions
# compiled_baleen_with_assertions = teleprompter.compile(
#     student=baleen_with_assertions,
#     teacher=baleen_with_assertions,
#     trainset=trainset,
#     valset=devset
# )

# 优化器
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps.
# The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=4)

teleprompter = dspy.BootstrapFewShotWithRandomSearch(
    metric=validate_context_and_answer_and_hops,
    max_bootstrapped_demos=max_bootstrapped_demos,
    num_candidate_programs=6,
)
optimized_program = teleprompter.compile(baleen_with_assertions, trainset=trainset)

optimized_program.save("C:\Users\xinjiang\Desktop\dspy", save_field_meta=True)

loaded_program = baleen_with_assertions()
loaded_program.load(path="C:\Users\xinjiang\Desktop\dspy")
