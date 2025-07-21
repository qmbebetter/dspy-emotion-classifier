import panel as pn  # GUI
import openai
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

pn.extension()

# 定义上下文
context = [{'role': 'system', 'content': """
You are OrderBot, an automated service to collect orders for a pizza restaurant.
You first greet the customer, then collect the order, 
and then ask if it's a pickup or delivery. 
You wait to collect the entire order, then summarize it and check for a final 
time if the customer wants to add anything else.
If it's a delivery, you ask for an address. 
Finally you collect the payment.
Make sure to clarify all options, extras and sizes to uniquely
identify the item from the menu.
You respond in a short, very conversational friendly style. 
The menu includes 
pepperoni pizza 12.95, 10.00, 7.00 
cheese pizza 10.95, 9.25, 6.50
eggplant pizza 11.95, 9.75, 6.75 
fries 4.50, 3.50 
greek salad 7.25 
Toppings:
extra cheese 2.00, 
mushrooms 1.50 
sausage 3.00 
canadian bacon 3.50 
AI sauce 1.50 
peppers 1.00 
Drinks:
coke 3.00, 2.00, 1.00 
sprite 3.00, 2.00, 1.00 
bottled water 5.00 
"""}]  # accumulate messages

# 用于显示的面板
panels = []


# 收集消息的函数
def collect_messages(event):
    user_input = inp.value  # 获取用户输入
    panels.append(pn.pane.Markdown(f"**User:** {user_input}"))  # 添加用户消息到面板
    response = generate_response(user_input)  # 生成订单机器人回复
    panels.append(pn.pane.Markdown(f"**OrderBot:** {response}"))  # 添加机器人回复到面板
    inp.value = ""  # 清空输入框
    update_dashboard()  # 更新仪表板


# 生成机器人回复的函数
def generate_response(user_input):
    # 将用户输入加入上下文
    context.append({"role": "user", "content": user_input})
    # 调用 OpenAI ChatCompletion API
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=context,
        temperature=0.7,
    )
    # 获取回复内容
    bot_reply = response.choices[0].message.content
    # 将机器人回复加入上下文
    context.append({"role": "assistant", "content": bot_reply})
    return bot_reply


# 更新仪表板以显示收集的消息
def update_dashboard():
    dashboard[2] = pn.Column(*panels)


# 创建文本输入和按钮
inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here...')
button_conversation = pn.widgets.Button(name="Chat!")

# 绑定按钮点击事件到函数
button_conversation.on_click(collect_messages)

# 创建仪表板
dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.Column(*panels)  # 显示消息面板
)

# 显示仪表板
dashboard.show()
