# from dspy.teleprompt import Teleprompter
#
#
# class LabeledFewShot(Teleprompter):
#     def __init__(self, k=16):
#         self.k = k
#
#     def compile(self, student, trainset):
#         pass
#
#
# import dspy
#
#
# # Define a simple signature for basic question answering
# class BasicQA(dspy.Signature):
#     """Answer questions with short factoid answers."""
#     question = dspy.InputField()
#     answer = dspy.OutputField(desc="often between 1 and 5 words")
#
#
# # Assume defined trainset
# class RAG(dspy.Module):
#     def __init__(self, num_passages=3):
#         super().__init__()
#
#         # declare retrieval and predictor modules
#         self.retrieve = dspy.Retrieve(k=num_passages)
#         self.generate_answer = dspy.ChainOfThought(BasicQA)
#
#     # flow for answering questions using predictor and retrieval modules
#     def forward(self, question):
#         context = self.retrieve(question).passages
#         prediction = self.generate_answer(context=context, question=question)
#         return dspy.Prediction(context=context, answer=prediction.answer)
#
#
# # Define teleprompter
# teleprompter = LabeledFewShot()
#
# from dspy.datasets import DataLoader
#
# dl = DataLoader()
#
# blog_alpaca = dl.from_huggingface(
#     "ruslanmv/ai-medical-chatbot",
#     input_keys=("title",)
# )
# train_split = blog_alpaca['train']
#
# # Compile!
# compiled_rag = teleprompter.compile(student=RAG(), trainset=train_split)
# print(compiled_rag)
import dspy
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 获取 OpenAI API 密钥
api_key = os.getenv("OPENAI_API_KEY")
import dspy
from dspy.datasets.gsm8k import gsm8k_metric

turbo = dspy.LM(model='openai/gpt-3.5-turbo', max_tokens=250)
dspy.configure(lm=turbo)

import random
from dspy.datasets import DataLoader

dl = DataLoader()

gsm8k = dl.from_huggingface(
    "openai/gsm8k",
    "main",
    input_keys=("question",),
)

trainset, devset = gsm8k['train'], random.sample(gsm8k['test'], 50)


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=devset[:], metric=gsm8k_metric, num_threads=6, display_progress=True,
                    display_table=False)
cot_baseline = CoT()

evaluate(cot_baseline)

from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=gsm8k_metric,
    max_bootstrapped_demos=8,
    max_labeled_demos=8,
    max_rounds=10,
)
cot_compiled = optimizer.compile(CoT(), trainset=trainset)

cot_compiled.save('turbo_gsm8k.json')

# Loading:
# cot = CoT()
# cot.load('turbo_gsm8k.json')
