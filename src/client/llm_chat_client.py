import os
import json
import re
from openai import OpenAI
from langchain_core.documents import Document


LLM_CHAT_PROMPT = """
### 信息
{context}

### 任务
你是上汽大众 Polo 系列车型的用户手册问答系统，你具备{{信息}}中的知识。
请回答问题"{query}"，答案需要精准，语句通顺，并严格按照以下格式输出

{{答案}}【{{引用编号1}}, {{引用编号2}}, ...】
如果无法从中得到答案，请说 "无答案" ，不允许在答案中添加编造成分。
"""


llm_client = OpenAI(
    api_key=os.environ['DOUBAO_API_KEY'],
    base_url=os.environ['DOUBAO_BASE_URL']
)


def request_chat(query, context):

    prompt = LLM_CHAT_PROMPT.format(context=context, query=query)

    completion = llm_client.chat.completions.create(
        model=os.environ["DOUBAO_MODEL_NAME"],
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096
    )
    result = completion.choices[0].message.content

    return result
