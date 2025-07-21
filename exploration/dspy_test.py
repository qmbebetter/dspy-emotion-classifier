# 相同的文件
#
import dspy
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 获取 OpenAI API 密钥
api_key = os.getenv("OPENAI_API_KEY")
# lm = dspy.LM('openai/gpt-3.5-turbo')
# dspy.configure(lm=lm)
#

# 适配器前面部分学习

# print(lm("hello! this is a raw prompt to GPT-4o-mini"))
#
# print(lm(messages=[{"role": "system", "content": "You are a helpful assistant."},
#              {"role": "user", "content": "What is 2+2?"}]))


# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
# qa = dspy.ChainOfThought('question -> answer')
# # Run with the default LM configured with `dspy.configure` above.
# response = qa(question="How many floors are in the castle David Gregory inherited?")
# print(response.answer)

# # Run with the default LM configured above, i.e. GPT-3.5
# qa = dspy.ChainOfThought('question -> answer')
# response = qa(question="How many floors are in the castle David Gregory inherited?")
# print('GPT-3.5:', response.answer)
#
# gpt4_turbo = dspy.LM('openai/gpt-4-1106-preview')#改dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=300)
#
# # Run with GPT-4 instead
# with dspy.context(lm=gpt4_turbo):
#     response = qa(question="How many floors are in the castle David Gregory inherited?")
#     print('GPT-4-turbo:', response.answer)

# gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)

# print(len(lm.history))  # e.g., 3 calls to the LM
#
# print(lm.history[-1].keys())  # access the last call to the LM, with all metadata

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm, experimental=True)

# fact_checking = dspy.ChainOfThought('claims -> verdicts')
# print(fact_checking(claims=["Python was released in 1991.", "Python is a compiled language."]))

# 签名部分
#
# sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.
#
# classify = dspy.Predict('sentence -> sentiment: bool')  # we'll see an example with Literal[] later
# var = classify(sentence=sentence).sentiment
# print(var)
#
# # Example from the XSum dataset.
# document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa
# League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League
# One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them
# from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest
# football transfers on our dedicated page."""
#
# summarize = dspy.ChainOfThought('document -> summary')
# response = summarize(document=document)
#
# # print(response.summary)
# print("Reasoning:", response.reasoning)
#
# # 更详细的签名
# from typing import Literal
# class Emotion(dspy.Signature):
#     """Classify emotion."""
#
#     sentence: str = dspy.InputField()
#     sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()
#
# sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion
#
# classify = dspy.Predict(Emotion)
# print(classify(sentence=sentence))
# class CheckCitationFaithfulness(dspy.Signature):
#     """Verify that the text is based on the provided context."""
#
#     context: str = dspy.InputField(desc="facts here are assumed to be true")
#     text: str = dspy.InputField()
#     faithfulness: bool = dspy.OutputField()
#     evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")
# context = ("The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa "
#            "League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells "
#            "in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was "
#            "unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been "
#            "revealed. Find all the latest football transfers on our dedicated page.")
# text = "Lee scored 3 goals for Colchester United."
# faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
# print(faithfulness(context=context, text=text))

# 数据

# qa_pair = dspy.Example(question="This is a question?", answer="This is an answer.")
#
# print(qa_pair)
# print(qa_pair.question)
# print(qa_pair.answer)

# article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article")
#
# input_key_only = article_summary.inputs()
# non_input_key_only = article_summary.labels()
#
# print("Example object with Input fields only:", input_key_only)
# print("Example object with Non-Input fields only:", non_input_key_only)

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
# # Since this is the only split in the dataset we can split this into
# # train and test split ourselves by slicing or sampling 75 rows from the train
# # split for testing.
# testset = train_split[:75]
# trainset = train_split

from dspy.datasets import DataLoader

dl = DataLoader()

code_alpaca = dl.from_huggingface("ruslanmv/ai-medical-chatbot")

train_dataset = code_alpaca['train']
print(train_dataset)
