import os
import dspy
import google.generativeai as genai
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class GeminiLM(dspy.LM):
    def __init__(self, model, api_key=None, endpoint=None, **kwargs):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY", api_key))

        self.endpoint = endpoint
        self.history = []

        super().__init__(model, **kwargs)
        self.model = genai.GenerativeModel(model)

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Custom chat model working for text completion model
        if messages:
            prompt = '\n\n'.join([x.get('content', '') for x in messages] + ['BEGIN RESPONSE:'])
        else:
            prompt = prompt or "Default Prompt Text"

        completions = self.model.generate_content(prompt)
        self.history.append({"prompt": prompt, "completions": completions})

        # Debug print to check completions structure
        print(completions)

        # Must return a list of strings
        try:
            return [completions.candidates[0].content.parts[0].text]
        except AttributeError:
            print("Error parsing completions response. Check structure.")
            return [""]

    def inspect_history(self):
        for interaction in self.history:
            print(f"Prompt: {interaction['prompt']} -> Completions: {interaction['completions']}")


# 测试 lm 实例
lm = GeminiLM("gemini-1.5-flash", temperature=0)
print(lm(prompt="What is the capital of France?"))

# 配置 dspy
dspy.configure(lm=lm)

# 搞不出来……好怪！
# 测试 ChainOfThought
qa = dspy.ChainOfThought("question->answer")
response = qa(question="What is the capital of France?")
print(response)
