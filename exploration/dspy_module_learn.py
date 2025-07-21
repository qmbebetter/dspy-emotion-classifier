# dspy模块学习

import os
import random

# 相同的文件
#
import dspy
from dotenv import load_dotenv
from dspy import Predict, ensure_signature, Module
from dspy.predict.parameter import Parameter

# 加载 .env 文件
load_dotenv()

# 获取 OpenAI API 密钥
api_key = os.getenv("OPENAI_API_KEY")

# lm = dspy.LM('openai/gpt-4o-mini')
# dspy.configure(lm=lm, experimental=True)

# sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.
#
# # 1) Declare with a signature.
# classify = dspy.Predict('sentence -> sentiment')
#
# # 2) Call with input argument(s).
# response = classify(sentence=sentence)
#
# # 3) Access the output.
# print(response.sentiment)

# question = "What's something great about the ColBERT retrieval model?"
#
# # 1) Declare with a signature, and pass some config.
# classify = dspy.ChainOfThought('question -> answer', n=5)
#
# # 2) Call with input argument.
# response = classify(question=question)
#
# # # 3) Access the outputs.
# # print(response.completions.answer)
#
# # print(f"Reasoning: {response.reasoning}")
# # print(f"Answer: {response.answer}")
#
# print(response.completions[3].reasoning == response.completions.reasoning[3])


# 预测
# class Predict(Parameter):
#     def __init__(self, signature, **config):
#         self.stage = random.randbytes(8).hex()
#         self.signature = signature
#         self.config = config
#         self.reset()
#
#         if isinstance(signature, str):
#             inputs, outputs = signature.split("->")
#             inputs, outputs = inputs.split(","), outputs.split(",")
#             inputs, outputs = [field.strip() for field in inputs], [field.strip() for field in outputs]
#
#             assert all(len(field.split()) == 1 for field in (inputs + outputs))
#
#             inputs_ = ', '.join([f"`{field}`" for field in inputs])
#             outputs_ = ', '.join([f"`{field}`" for field in outputs])
#
#             instructions = f"""Given the fields {inputs_}, produce the fields {outputs_}."""
#
#             inputs = {k: InputField() for k in inputs}
#             outputs = {k: OutputField() for k in outputs}
#
#             for k, v in inputs.items():
#                 v.finalize(k, infer_prefix(k))
#
#             for k, v in outputs.items():
#                 v.finalize(k, infer_prefix(k))
#
#             self.signature = dsp.Template(instructions, **inputs, **outputs)
#
#
# # Define a simple signature for basic question answering
# class BasicQA(dspy.Signature):
#     """Answer questions with short factoid answers."""
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 1 and 5 words")
#
#
# # Pass signature to Predict module
# generate_answer = dspy.Predict(BasicQA)
#
# # Call the predictor on a particular input.
# question = 'What is the color of the sky?'
# pred = generate_answer(question=question)
#
# print(f"Question: {question}")
# print(f"Predicted Answer: {pred.answer}")
########

from pydantic import BaseModel, Field


# class Input(BaseModel):
#     context: str = Field(description="The context for the question")
#     query: str = Field(description="The question to be answered")
#
#
# class Output(BaseModel):
#     answer: str = Field(description="The answer for the question")
#     confidence: float = Field(ge=0, le=1, description="The confidence score for the answer")
#
#
# class QASignature(dspy.Signature):
#     """Answer the question based on the context and query provided, and on the scale of 10 tell how confident you are
#     about the answer."""
#
#     input: Input = dspy.InputField()
#     output: Output = dspy.OutputField()


# predictor = dspy.TypedPredictor(QASignature)
#
# predictor = dspy.TypedPredictor('input -> output')  # 行不通不知道为啥……
#
# doc_query_pair = Input(
#     context="The quick brown fox jumps over the lazy dog",
#     query="What does the fox jumps over?",
# )
#
# prediction = predictor(input=doc_query_pair)
#
# answer = prediction.output.answer
# confidence_score = prediction.output.confidence
#
# print(f"Prediction: {prediction}\n\n")
# print(f"Answer: {answer}, Answer Type: {type(answer)}")
# print(f"Confidence Score: {confidence_score}, Confidence Score Type: {type(confidence_score)}")

# # 换成TypedChainOfThought
# cot_predictor = dspy.TypedChainOfThought(QASignature)
#
# doc_query_pair = Input(
#     context="The quick brown fox jumps over the lazy dog",
#     query="What does the fox jumps over?",
# )


# prediction = cot_predictor(input=doc_query_pair)


# answer = prediction.output.answer
# confidence_score = prediction.output.confidence
#
# print(f"Prediction: {prediction}\n\n")
# print(f"Answer: {answer}, Answer Type: {type(answer)}")
# print(f"Confidence Score: {confidence_score}, Confidence Score Type: {type(confidence_score)}")

# @dspy.predictor
# def answer(doc_query_pair: Input) -> Output:
#     """Answer the question based on the context and query provided, and on the scale of 0-1 tell how confident you
#     are about the answer."""
#     pass
#
#
# @dspy.cot
# def answer(doc_query_pair: Input) -> Output:
#     """Answer the question based on the context and query provided, and on the scale of 0-1 tell how confident you
#     are about the answer."""
#     pass
#
#
# prediction = answer(doc_query_pair=doc_query_pair)
#
# print(f"Prediction: {prediction}\n\n")


# # 组合函数类型预测器
#
# # 定义去重函数
# def deduplicate(items):
#     """去重函数，保持顺序"""
#     return list(dict.fromkeys(items))
#
# # 定义SimplifiedBaleen类
# class SimplifiedBaleen(dspy.Module):
#     def __init__(self, passages_per_hop=3, max_hops=1):
#         """
#         初始化SimplifiedBaleen类
#         :param passages_per_hop: 每次跳跃检索的段落数
#         :param max_hops: 最大跳跃次数
#         """
#         super().__init__()
#         self.retrieve = dspy.Retrieve(k=passages_per_hop)  # 设置检索对象
#         self.max_hops = max_hops  # 设置最大跳跃次数
#
#     @dspy.cot  # 定义思路链装饰器
#     def generate_query(self, context: list[str], question) -> str:
#         """生成简单的检索查询"""
#         return f"Search query for: {question}"  # 返回检索查询语句
#
#     @dspy.cot  # 定义思路链装饰器
#     def generate_answer(self, context: list[str], question) -> str:
#         """从上下文中生成简短答案"""
#         return f"Answer based on context: {' '.join(context)}"  # 返回基于上下文的答案
#
#     def forward(self, question: str) -> dspy.Prediction:
#         """
#         使用迭代检索和推理来回答复杂问题
#         :param question: 用户提出的问题
#         :return: dspy.Prediction对象，包含上下文和答案
#         """
#         context = []  # 初始化上下文
#
#         for _ in range(self.max_hops):  # 根据最大跳跃次数进行循环
#             query = self.generate_query(context=context, question=question)  # 生成查询
#             retrieved_result = self.retrieve(query)  # 检索相关段落
#             passages = retrieved_result.passages  # 获取检索到的段落
#             context = deduplicate(context + passages)  # 合并去重上下文和段落
#
#             answer = self.generate_answer(context=context, question=question)  # 基于上下文生成答案
#
#         return dspy.Prediction(context=context, answer=answer)  # 返回预测结果
#
#
# # 创建SimplifiedBaleen类的实例
# cot_predictor = SimplifiedBaleen(passages_per_hop=3, max_hops=1)
#
# # 输入问题
# question = "What does the fox jump over?"
#
# # 调用forward方法进行推理
# prediction = cot_predictor.forward(question)
#
# # 打印预测结果
# print(f"Prediction Context: {prediction.context}")
# print(f"Answer: {prediction.answer}")


import dspy
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.teleprompt.signature_opt_typed import optimize_signature

# 配置语言模型
gpt4 = dspy.LM('openai/gpt-4', max_tokens=250)


# 定义 QASignature 签名类
class QASignature(dspy.Signature):
    """
    Defines the input and output fields for a QA task.
    """
    answer: str = dspy.OutputField()


# 定义评估器
evaluator = Evaluate(
    devset=devset,
    metric=answer_exact_match,
    num_threads=10,  # 设定多线程
    display_progress=True
)

# 优化预测器
result = optimize_signature(
    student=dspy.TypedPredictor(QASignature),  # 使用定义的签名
    evaluator=evaluator,  # 评估器
    initial_prompts=6,  # 初始生成的提示
    n_iterations=50,  # 最大迭代次数（可减少避免重复）
    max_examples=20,  # 每轮最大样例数（减少样例以简化优化）
    verbose=True,  # 输出详细优化过程
    prompt_model=gpt4,  # 使用 GPT-4 作为提示优化模型
)

# 打印优化结果
print("Optimization completed!")
print(result)


# class ChainOfThought(Predict):
#     def __init__(self, signature, rationale_type=None, activated=True, **config):
#         super().__init__(signature, **config)
#
#         self.activated = activated
#
#         self.signature = signature = ensure_signature(signature)
#         *_keys, last_key = signature.output_fields.keys()
#
#         prefix = "Reasoning: Let's think step by step in order to"  # 用于引导推理过程
#
#         if isinstance(dspy.settings.lm, dspy.LM):
#             desc = "${reasoning}"
#         elif dspy.settings.experimental:  # 实验
#             desc = "${produce the output fields}. We ..."
#         else:
#             # For dspy <2.5 旧版本
#             desc = f"${{produce the {last_key}}}. We ..."
#
#         rationale_type = rationale_type or dspy.OutputField(prefix=prefix, desc=desc)  # desc用来设置推理过程的输出字段格式
#
#         # Add "rationale" field to the output signature.
#         if isinstance(dspy.settings.lm, dspy.LM) or dspy.settings.experimental:
#             extended_signature = signature.prepend("reasoning", rationale_type, type_=str)
#         else:
#             # For dspy <2.5
#             extended_signature = signature.prepend("rationale", rationale_type, type_=str)
#
#         self._predict = dspy.Predict(extended_signature, **config)
#         self._predict.extended_signature = extended_signature


# # Define a simple signature for basic question answering 定义一个简单的签名对象
# class BasicQA(dspy.Signature):
#     """Answer questions with short factoid answers."""
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 1 and 5 words")
#
#
# # define a custom rationale
# rationale_type = dspy.OutputField(
#     prefix="Reasoning: Let's think step by step in order to",
#     desc="${produce the answer}. We ...",
# )
# # Pass signature to ChainOfThought module
# generate_answer = dspy.ChainOfThought(BasicQA, rationale_type=rationale_type)
#
# # Call the predictor on a particular input.
# question = '1 plus 1 equal to'
# pred = generate_answer(question=question)
#
# print(f"Question: {question}")
# print(f"Predicted Answer: {pred.answer}")


# class BasicQA(dspy.Signature):
#     """Answer questions with short factoid answers."""
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 1 and 5 words")
#
#
# # Pass signature to ChainOfThought module
# generate_answer = dspy.ChainOfThoughtWithHint(BasicQA)
#
# # Call the predictor on a particular input alongside a hint.
# question = 'What is the color of the sky?'
# hint = "It's what you often see during a sunny day."
# pred = generate_answer(question=question, hint=hint)
#
# print(f"Question: {question}")
# print(f"Predicted Answer: {pred.answer}")


# # Define a simple signature for basic question answering
# class BasicQA(dspy.Signature):
#     """Answer questions with short factoid answers."""
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 1 and 5 words")
#
#
# # Pass signature to ReAct module
# react_module = dspy.ReAct(BasicQA)
#
# # Call the ReAct module on a particular input
# question = ('Aside from the Apple Remote, what other devices can control the program Apple Remote was originally '
#             'designed to interact with?')
# result = react_module(question=question)
#
# # print(f"Question: {question}")
# # print(f"Final Predicted Answer (after ReAct process): {result.answer}")
#
# lm.inspect_history(n=3)

# # dspy.多链比较
# import dsp
#
#
# class MultiChainComparison(Module):
#     def __init__(self, signature, M=3, temperature=0.7, **config):
#         super().__init__()
#
#         self.M = M
#         signature = Predict(signature).signature
#         *keys, last_key = signature.kwargs.keys()
#
#         extended_kwargs = {key: signature.kwargs[key] for key in keys}
#
#         for idx in range(M):
#             candidate_type = dsp.Type(prefix=f"Student Attempt #{idx + 1}:", desc="${reasoning attempt}")
#             extended_kwargs.update({f'reasoning_attempt_{idx + 1}': candidate_type})
#
#         rationale_type = dsp.Type(prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
#                                   desc="${corrected reasoning}")
#         # 这样做是为了为每次推理尝试保留原始签名中的所有字段（不包括最后一个字段）。
#         extended_kwargs.update({'rationale': rationale_type, last_key: signature.kwargs[last_key]})
#
#         signature = dsp.Template(signature.instructions, **extended_kwargs)
#         self.predict = Predict(signature, temperature=temperature, **config)
#         self.last_key = last_key
#
#
# class BasicQA(dspy.Signature):
#     """Answer questions with short factoid answers."""
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 1 and 5 words")
#
#
# # Example completions generated by a model for reference
# completions = [
#     dspy.Prediction(rationale="I recall that during clear days, the sky often appears this color.", answer="blue"),
#     dspy.Prediction(rationale="Based on common knowledge, I believe the sky is typically seen as this color.",
#                     answer="green"),
#     dspy.Prediction(rationale="From images and depictions in media, the sky is frequently represented with this hue.",
#                     answer="blue"),
# ]
#
# # Pass signature to MultiChainComparison module
# compare_answers = dspy.MultiChainComparison(BasicQA)
#
# # Call the MultiChainComparison on the completions
# question = 'What is the color of the sky?'
# final_pred = compare_answers(completions, question=question)
#
# print(f"Question: {question}")
# print(f"Final Predicted Answer (after comparison): {final_pred.answer}")
# print(f"Final Rationale: {final_pred.rationale}")


# # 思维程序 (PoT) 提示技术
# # Define a simple signature for basic question answering
# class BasicQA(dspy.Signature):
#     """Answer questions with short factoid answers."""
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 1 and 5 words")
#
#
# # Pass signature to ProgramOfThought Module
# pot = dspy.ProgramOfThought(BasicQA)
#
# #Call the ProgramOfThought module on a particular input
# question = 'Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?'
# result = pot(question=question)
#
# # print(f"Question: {question}")
# # print(f"Final Predicted Answer (after ProgramOfThought process): {result.answer}")
#
# lm.inspect_history(n=4)

# class Retrieve(Parameter):
#     def __init__(self, k=3):
#         self.stage = random.randbytes(8).hex()
#         self.k = k
#
#
# # Example Usage
#
# # Define a retrieval model server to send retrieval requests to
# colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
#
# # Configure retrieval server internally
# dspy.settings.configure(rm=colbertv2_wiki17_abstracts)
#
# # Define Retrieve Module
# retriever = dspy.Retrieve(k=3)
#
# query='When was the first FIFA World Cup held?'
#
# # Call the retriever on a particular query.
# topK_passages = retriever(query).passages
#
# print(f"Top {retriever.k} passages for question: {query} \n", '-' * 30, '\n')
#
# for idx, passage in enumerate(topK_passages):
#     print(f'{idx+1}]', passage, '\n')


# sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.
#
# # 1) Declare with a signature.
# classify = dspy.Predict('sentence -> sentiment')
#
# # 2) Call with input argument(s).
# response = classify(sentence=sentence)
#
# # 3) Access the output.
# print(response.sentiment)

# question = "What's something great about the ColBERT retrieval model?"
#
# # 1) Declare with a signature, and pass some config.
# classify = dspy.ChainOfThought('question -> answer', n=5)
#
# # 2) Call with input argument.
# response = classify(question=question)
#
# # # 3) Access the outputs.
# # print(response.completions.answer)
#
# # print(f"Reasoning: {response.reasoning}")
# # print(f"Answer: {response.answer}")
#
# print(response.completions[3].reasoning == response.completions.reasoning[3])
