# 提示词如何写学习
import openai
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')


def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


# text = f"""
# You should express what you want a model to do by \
#     providing instructions that are as clear and \
#     specific as you can possibly make them. \
#     This will guide the model towards the desired output,\
#     and reduce the chances of receiving irrelevant \
#     or incorrect responses. Don't confuse writing a \
#     clear prompt with writing a short prompt.\
#     In many cases, longer prompts provide more clarity \
#     and context for the model, which can lead to \
#     more detailed and relevant outputs.'
#     """
# prompt =f"""
#     Summarize the text delimited by triple backticks \
#     into a single sentence.
#     '''{text}'''
#      """
# print(response)

# prompt =f"""
#     Generate a list of three made-up book titles along \
#     with their authors and genres.
#     Provide them in JSON format with the following keys:
#     book_id, title, author, genre.
#     """
# response = get_completion(prompt)
# print(response)

# text_1= f"""
# Making a cup of tea is easy! First, you need to get some \
# water boiling. While that's happening, \
# grab a cup and put a tea bag in it. Once the water is \
# hot enough, just pour it over the tea bag. \
# Let it sit for a bit so the tea can steep. After a \
# few minutes, take out the tea bag. If you \
# like, you can add some sugar or milk to taste. \
# And that's it! You've got yourself a delicious \
# cup of tea to enjoy.
# """
# prompt = f"""
# You will be provided with text delimited by triple quotes.
# If it contains a sequence of instructions, \
# re-write those instructions in the following format:
# Step 1-..
# Step 2 -
# …
# Step N - …
# If the text does not contain a sequence of instructions, \
# then simply write \"No steps provided.\"
# \"\"\"{text_1}\"\"\"
# """
# response = get_completion(prompt)
# print("Completion for Text 1:")
# print(response)

# text_2 = f"""
# The sun is shining brightly today, and the birds are \
# singing. It's a beautiful day to go for a \
# walk in the park. The flowers are blooming, and the \
# trees are swaying gently in the breeze. People \
# are out and about, enjoying the lovely weather.\
# Some are having picnics, while others are playing \
# games or simply relaxing on the grass. It's a \
# perfect day to spend time outdoors and appreciate the\
#  beauty of nature.
# """
# prompt = f"""
# You will be provided with text delimited by triple quotes.
# If it contains a sequence of instructions, \
# re-write those instructions in the following format:
# Step 1-..
# Step 2 -
# …
# Step N - …
# If the text does not contain a sequence of instructions, \
# then simply write \"No steps provided.\"
# \"\"\"{text_2}\"\"\"
# """
# response = get_completion(prompt)
# print("Completion for Text 2:")
# print(response)

# prompt=f"""
# Your task is to answer in a consistent style.
# < child >: Teach me about patience.
# < grandparent >: The river that carves the deepest \
# valley flows from a modest spring;the \
# grandest symphony originates from a single note; \
# the most intricate tapestry begins with a solitary thread.
# < child >: Teach me about resilience.
# """
# response = get_completion(prompt)
# print(response)

# text = f"""
# In a charming village, siblings Jack and Jill set out on \
# a quest to fetch water from a hilltop \
# well. As they climbed, singing joyfully, misfortune\
# struck-Jack tripped on a stone and tumbled \
# down the hill, with Jill following suit. \
# Though slightly battered, the pair returned home to \
# comforting embraces. Despite the mishap, \
# their adventurous spirits remained undimmed, and they \
# continued exploring with delight.
# """
# # example 1
# prompt_1 = f"""
# Perform the following actions:
# 1 - Summarize the following text delimited by triple \
# backticks with 1 sentence.
# 2 - Translate the summary into French.
# 3 - List each name in the French summary.
# 4 - Output a json object that contains the following \
# keys: french_summary, num _names.
# Separate your answers with line breaks.
# Text:
# '''{text}'''
# """
# response = get_completion(prompt_1)
# print("Completion for prompt 1:")
# print(response)

#
# # example 2
# prompt_2 = f"""
# Perform the following actions:
# 1 - Summarize the following text delimited by <> with 1 sentence.
# 2 - Translate the summary into French.
# 3 - List each name in the French summary.
# 4 - Output a json object that contains the following \
# keys: french_summary, num _names.
# Use the following format:
# Text:<text to summarize>
# Summary: <summary>
# Translation: <summary translation>
# Names: <list of names in Italian summary>
# Output JSON: <json with summary and num_names>
# Text:<{text}>
# """
# response = get_completion(prompt_2)
# print("\nCompletion for prompt 2:")
# print(response)

# user_messages = [
#     "La performance du système est plus lente que d'habitud ",
#     "Mi monitor tiene píxeles que no se iluminan.",
#     "Il mio mouse non funziona",
#     "Mój klawisz Ctrl jest zepsuty",
#     "我的屏幕在闪烁"
# ]
# for issue in user_messages:
#     prompt = f"Tell me what language this is: '''{issue}''' Use only one word to give the answer"
#     lang = get_completion(prompt)
#     print(f"Original message ({lang}): {issue}")
#     prompt = f"""
#     Translate the following text to English \
#     and Korean:'''{issue}'''
#     """
#     response = get_completion(prompt, temperature=0.7)
#     print(response, "\n")
