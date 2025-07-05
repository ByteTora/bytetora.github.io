"""
命题分块 核心函数
"""
import re
import os
import json
import fitz
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
# 命题生成 完整的流程
####################################
def process_document_into_propositions(pdf_path, chunk_size=800, chunk_overlap=100, quality_thresholds=None):
    """
    将文档处理为经过质量检查的命题。

    Args:
        pdf_path (str): PDF文件的路径
        chunk_size (int): 每个分块的字符大小
        chunk_overlap (int): 分块之间的重叠字符数
        quality_thresholds (Dict): 命题质量的阈值分数

    Returns:
        Tuple[List[Dict], List[Dict]]: 原始分块和命题分块
    """
    # 如果未提供，则设置默认的质量阈值
    if quality_thresholds is None:
        quality_thresholds = {
            "accuracy": 7,  # 准确性阈值
            "clarity": 7,  # 清晰性阈值
            "completeness": 7,  # 完整性阈值
            "conciseness": 7  # 简洁性阈值
        }

    # 从PDF文件中提取文本
    text = extract_text_from_pdf(pdf_path)

    # 从提取的文本创建分块
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # 初始化一个列表以存储所有命题
    all_propositions = []

    print("从分块生成命题...")
    for i, chunk in enumerate(chunks):
        print(f"处理分块 {i + 1}/{len(chunks)}...")

        # 为当前分块生成命题
        chunk_propositions = generate_propositions(chunk)
        print(f"生成了 {len(chunk_propositions)} 个命题")

        # 处理每个生成的命题
        for prop in chunk_propositions:
            proposition_data = {
                "text": prop,  # 命题文本
                "source_chunk_id": chunk["chunk_id"],  # 来源分块ID
                "source_text": chunk["text"]  # 来源分块文本
            }
            all_propositions.append(proposition_data)

    # 评估生成的命题质量
    print("\n评估命题质量...")
    quality_propositions = []

    for i, prop in enumerate(all_propositions):
        if i % 10 == 0:  # 每10个命题进行一次状态更新
            print(f"评估命题 {i + 1}/{len(all_propositions)}...")

        # 评估当前命题的质量
        scores = evaluate_proposition(prop["text"], prop["source_text"])
        prop["quality_scores"] = scores

        # 检查命题是否通过质量阈值
        passes_quality = True
        for metric, threshold in quality_thresholds.items():
            if scores.get(metric, 0) < threshold:
                passes_quality = False
                break

        if passes_quality:
            quality_propositions.append(prop)
        else:
            print(f"命题未通过质量检查: {prop['text'][:50]}...")

    print(f"\n在质量过滤后保留了 {len(quality_propositions)}/{len(all_propositions)} 个命题")

    return chunks, quality_propositions


####################################
# 命题生成
####################################
def generate_propositions(chunk):
    """
    从文本块中生成原子化、自包含的命题。

    Args:
        chunk (Dict): 包含内容和元数据的文本块

    Returns:
        List[str]: 生成的命题列表
    """
    # 系统提示，指示AI如何生成命题
    system_prompt = """请将以下文本分解为简单的自包含命题。确保每个命题符合以下标准：

    1. 表达单一事实：每个命题应陈述一个具体事实或主张
    2. 独立可理解：命题应自成体系，无需额外上下文即可理解
    3. 使用全称而非代词：避免使用代词或模糊指代，使用完整的实体名称
    4. 包含相关日期/限定条件：如适用应包含必要日期、时间和限定条件以保持准确性
    5. 保持单一主谓关系：聚焦单个主体及其对应动作或属性，避免连接词和多从句结构

    请仅输出命题列表，不要包含任何额外文本或解释。
    """
    # 用户提示，包含要转换为命题的文本块
    user_prompt = f"要转换为命题的文本:\n\n{chunk['text']}"

    # 从模型生成响应
    response = client.chat.completions.create(
        model=llm_model,  # 使用更强的模型以准确生成命题
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从响应中提取命题
    raw_propositions = response.choices[0].message.content.strip().split('\n')

    # 清理命题（移除编号、项目符号等）
    clean_propositions = []
    for prop in raw_propositions:
        # 移除编号（如1., 2.等）和项目符号
        cleaned = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', prop).strip()
        if cleaned and len(cleaned) > 10:  # 简单过滤空或过短的命题
            clean_propositions.append(cleaned)

    return clean_propositions


####################################
# 评估命题质量
####################################
def evaluate_proposition(proposition, original_text):
    """
    根据准确性、清晰性、完整性以及简洁性评估命题的质量。

    Args:
        proposition (str): 要评估的命题
        original_text (str): 用于比较的原文

    Returns:
        Dict: 每个评估维度的分数
    """
    # 系统提示，指示AI如何评估命题
    system_prompt = """你是一位评估从文本中提取命题质量的专家。请根据以下标准对给定命题进行评分（1-10分）：

    - 准确性（Accuracy）：命题反映原文信息的准确程度
    - 清晰性（Clarity）：不依赖额外上下文的情况下，命题是否易于理解
    - 完整性（Completeness）：命题是否包含必要的细节（如日期、限定词等）
    - 简洁性（Conciseness）：命题是否在保留关键信息前提下，表述精简程度

    响应必须为有效的JSON格式，并包含每个标准的数值评分：
    {"accuracy": X, "clarity": X, "completeness": X, "conciseness": X}
    """

    # 用户提示，包含命题和原文
    user_prompt = f"""命题: {proposition}

    原文: {original_text}

    请以JSON格式提供你的评分。"""

    # 从模型生成响应
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    # 解析JSON响应
    try:
        scores = json.loads(response.choices[0].message.content.strip())
        return scores
    except json.JSONDecodeError:
        # 如果JSON解析失败，使用默认值作为回退
        return {
            "accuracy": 5,
            "clarity": 5,
            "completeness": 5,
            "conciseness": 5
        }


def extract_text_from_pdf(pdf_path):
    """
    从 PDF 文件中提取文本，并打印前 `num_chars` 个字符。

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # 打开 PDF 文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串以存储提取的文本

    # Iterate through each page in the PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")  # 从页面中提取文本
        all_text += text  # 将提取的文本追加到 all_text 字符串中

    return all_text  # 返回提取的文本


def chunk_text(text, chunk_size=800, overlap=100):
    """
    将文本分割为重叠的块。

    Args:
        text (str): 要分割的输入文本
        chunk_size (int): 每个块的字符数
        overlap (int): 块之间的字符重叠数

    Returns:
        List[Dict]: 包含文本和元数据的块字典列表
    """
    chunks = []  # 初始化一个空列表来存储块

    # 使用指定的块大小和重叠迭代文本
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]  # 提取指定大小的块
        if chunk:  # 确保不添加空块
            chunks.append({
                "text": chunk,  # 块文本
                "chunk_id": len(chunks) + 1,  # 块的唯一ID
                "start_char": i,  # 块的起始字符索引
                "end_char": i + len(chunk)  # 块的结束字符索引
            })

    print(f"创建了 {len(chunks)} 个文本块")  # 打印创建的块数
    return chunks  # 返回块列表
