"""
文档增强：问题生成 核心函数
"""
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
llm_model = os.getenv("LLM_MODEL_ID")
embedding_model = os.getenv("EMBEDDING_MODEL_ID")

####################################
# 文档增强：问题生成，根据文本块语义生成相关问题
####################################
def generate_questions(text_chunk, num_questions=5):
    """
    生成可以从给定文本块中回答的相关问题。

    Args:
        text_chunk (str): 要生成问题的文本块。
        num_questions (int): 要生成的问题数量。
        model (str): 用于生成问题的模型。

    Returns:
        List[str]: 生成的问题列表。
    """
    # 定义系统提示
    system_prompt = "你是一个从文本中生成相关问题的专家。能够根据用户提供的文本生成可回答的简洁问题，重点聚焦核心信息和关键概念。"

    # 定义用户提示，包含文本块和要生成的问题数量
    # user_prompt = f"""
    # 根据以下文本，生成 {num_questions} 个不同的问题，这些问题只能通过此文本回答：
    #
    # {text_chunk}
    #
    # 请以编号列表的形式回复问题，且不要添加任何额外文本。
    # """
    user_prompt = f"""
    请根据以下文本内容生成{num_questions}个不同的、仅能通过该文本内容回答的问题：

    {text_chunk}

    请严格按以下格式回复：
    1. 带编号的问题列表
    2. 仅包含问题
    3. 不要添加任何其他内容
    """

    # 使用 OpenAI API 生成问题
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 从响应中提取并清理问题
    questions_text = response.choices[0].message.content.strip()

    # 使用正则表达式模式匹配提取问题
    pattern = r'^\d+\.\s*(.*)'
    return [re.match(pattern, line).group(1) for line in questions_text.split('\n') if line.strip()]
    # 此处改变了原有的正则处理，避免生成的问题中没有问号以及中英文问号匹配的问题
    # questions = []
    # for line in questions_text.split('\n'):
    #     # 去除编号并清理空白
    #     cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip())
    #     # if cleaned_line and cleaned_line.endswith('？') or cleaned_line.endswith("?"):
    #     #     questions.append(cleaned_line)
    #
    # return questions






