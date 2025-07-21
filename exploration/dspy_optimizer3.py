# MIPROv2优化器
# 成功示例！哈哈哈

import dspy
from dotenv import load_dotenv
import os

from dspy import cot

# 加载 .env 文件
load_dotenv()

# 获取 OpenAI API 密钥
api_key = os.getenv("OPENAI_API_KEY")

import dspy

turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=250)
dspy.settings.configure(lm=turbo)

from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

gms8k = GSM8K()

trainset, devset = gms8k.train, gms8k.dev


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=devset[:], metric=gsm8k_metric, num_threads=8, display_progress=True, display_table=False)

program = CoT()

evaluate(program, devset=devset[:])

# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light",  # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing program with MIPRO...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    requires_permission_to_run=False,
)

# Save optimize program for future use
optimized_program.save(f"mipro_optimized")

# Evaluate optimized program
print(f"Evaluate optimized program...")
evaluate(optimized_program, devset=devset[:])

# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light",  # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing zero-shot program with MIPRO...")
zeroshot_optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=0,  # ZERO FEW-SHOT EXAMPLES
    max_labeled_demos=0,  # ZERO FEW-SHOT EXAMPLES
    requires_permission_to_run=False,
)

# Save optimize program for future use
zeroshot_optimized_program.save(f"mipro_zeroshot_optimized")

# Evaluate optimized program
print(f"Evaluate optimized program...")
evaluate(zeroshot_optimized_program, devset=devset[:])
