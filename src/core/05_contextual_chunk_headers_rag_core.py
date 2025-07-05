"""
上下文标题块 核心函数
"""
import os
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
# 将文本切分成块并为每块生成标题
####################################
def chunk_text_with_headers(text, n, overlap):
    """
    将文本分割为较小的片段，并生成标题。

    Args:
        text (str): 要分块的完整文本
        n (int): 每个块的字符数
        overlap (int): 块之间的重叠字符数

    Returns:
        List[dict]: 包含 'header' 和 'text' 键的字典列表
    """
    chunks = []

    # 按指定的块大小和重叠量遍历文本
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]
        header = generate_chunk_header(chunk)  # 使用 LLM 为块生成标题
        chunks.append({"header": header, "text": chunk})  # 将标题和块添加到列表中

    return chunks


####################################
# 使用LLM 为块生成标题
####################################
def generate_chunk_header(chunk):
    """
    使用 LLM 为给定的文本块生成标题/页眉

    Args:
        chunk (str): T要总结为标题的文本块
        model (str): 用于生成标题的模型

    Returns:
        str: 生成的标题/页眉
    """
    # 定义系统提示
    system_prompt = "为给定的文本生成一个简洁且信息丰富的标题。"

    # 根据系统提示和文本块生成
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
    )

    # 返回生成的标题/页眉，去除任何前导或尾随空格
    return response.choices[0].message.content.strip()
