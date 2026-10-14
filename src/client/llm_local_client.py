import os
import json
import re
from openai import OpenAI
from langchain_core.documents import Document
from src.constant import qwen3_8b_tune_model_name


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
    api_key="EMPTY",
    base_url=os.environ.get("RAG_LLM_BASE_URL", "http://localhost:8000/v1")
)


def request_chat(query, context, stream=False):

    prompt = LLM_CHAT_PROMPT.format(context=context, query=query)

    completion = llm_client.chat.completions.create(
        model=qwen3_8b_tune_model_name,
        messages=[
            {"role": "system", "content": "你是一个有用的人工智能助手."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        frequency_penalty=2.0,
        temperature=0.001,
        top_p=0.95,
        stream=stream,
        extra_body={
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )
    if not stream:
        result = completion.choices[0].message.content
    else:
        result = completion

    return result
