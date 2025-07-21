###
# 自定义适配器部分学习
###
import dspy
from dspy.adapters.base import Adapter
from typing import List, Dict

from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 获取 OpenAI API 密钥
api_key = os.getenv("OPENAI_API_KEY")


class UpperCaseAdapter(Adapter):
    def __init__(self):
        super().__init__()

    # 格式化输入
    def format(self, signature, demos, inputs):
        system_prompt = signature.instructions
        all_fields = signature.model_fields
        all_field_data = [(all_fields[f].json_schema_extra["prefix"], all_fields[f].json_schema_extra["desc"]) for f in
                          all_fields]

        all_field_data_str = "\n".join([f"{p}: {d}" for p, d in all_field_data])
        format_instruction_prompt = "=" * 20 + f"""\n\nOutput Format:\n\n{all_field_data_str}\n\n""" + "=" * 20

        all_input_fields = signature.input_fields
        input_fields_data = [(all_input_fields[f].json_schema_extra["prefix"], inputs[f]) for f in all_input_fields]

        input_fields_str = "\n".join([f"{p}: {v}" for p, v in input_fields_data])

        # Convert to uppercase
        return (system_prompt + format_instruction_prompt + input_fields_str).upper()

    # 解析输出
    def parse(self, signature, completions, _parse_values=None):
        output_fields = signature.output_fields

        output_dict = {}
        for field in output_fields:
            field_info = output_fields[field]
            prefix = field_info.json_schema_extra["prefix"]

            field_completion = completions.split(prefix.upper())[-1].split("\n")[0].strip(": ")
            output_dict[field] = field_completion

        return output_dict


dspy.configure(adapter=UpperCaseAdapter())
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm, adapter=UpperCaseAdapter())

qa = dspy.ChainOfThought('question -> answer')

response = qa(question="How many floors are in the castle David Gregory inherited?")

lm.inspect_history()
