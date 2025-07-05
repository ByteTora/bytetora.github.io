"""
查询转换 核心函数
"""
import os
import re
import fitz
import numpy as np
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
# 查询转换完整流程
####################################
def rag_with_query_transformation(pdf_path, query, transformation_type=None):
    """
    运行完整的RAG管道，并可选地进行查询转换。

    Args:
        pdf_path (str): PDF文档的路径
        query (str): 用户查询
        transformation_type (str): 转换类型（None、'rewrite'、'step_back' 或 'decompose'）

    Returns:
        Dict: 包括原始查询、转换后的查询、上下文和回答的结果
    """
    # 处理文档以创建向量存储：从PDF 中提取文本，分割文本块并创建向量存储
    vector_store = process_document(pdf_path)

    # 应用查询转换并搜索
    if transformation_type:
        # 使用转换后的查询进行搜索
        results = transformed_search(query, vector_store, transformation_type)
    else:
        # 不进行转换，执行常规搜索
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=3)

    # 从搜索结果中组合上下文
    context = "\n\n".join([f"段落 {i + 1}:\n{result['text']}" for i, result in enumerate(results)])

    # 根据查询和组合后的上下文生成响应
    response = generate_response(query, context)

    # 返回结果，包括原始查询、转换类型、上下文和响应
    return {
        "original_query": query,
        "transformation_type": transformation_type,
        "context": context,
        "response": response
    }


####################################
# 使用查询转换进行文档检索
####################################
def transformed_search(query, vector_store, transformation_type, top_k=3):
    """
    使用转换后的查询进行搜索。

    Args:
        query (str): 原始查询
        vector_store (SimpleVectorStore): 用于搜索的向量存储
        transformation_type (str): 转换类型 ('rewrite', 'step_back', 或 'decompose')
        top_k (int): 返回的结果数量

    Returns:
        List[Dict]: 搜索结果
    """
    print(f"转换类型: {transformation_type}")
    print(f"原始查询: {query}")

    results = []

    if transformation_type == "rewrite":
        # 查询重写
        transformed_query = rewrite_query(query)
        print(f"重写后的查询: {transformed_query}")

        # 为转换后的查询创建嵌入向量
        query_embedding = create_embeddings(transformed_query)

        # 使用重写后的查询进行搜索
        results = vector_store.similarity_search(query_embedding, k=top_k)

    elif transformation_type == "step_back":
        # 回退提示
        transformed_query = generate_step_back_query(query)
        print(f"后退查询: {transformed_query}")

        # 为转换后的查询创建嵌入向量
        query_embedding = create_embeddings(transformed_query)

        # 使用回退查询进行搜索
        results = vector_store.similarity_search(query_embedding, k=top_k)

    elif transformation_type == "decompose":
        # 子查询分解
        sub_queries = decompose_query(query)
        print("分解为子查询:")
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"{i}. {sub_q}")

        # 为所有子查询创建嵌入向量
        sub_query_embeddings = create_embeddings(sub_queries)

        # 使用每个子查询进行搜索并合并结果
        all_results = []
        for i, embedding in enumerate(sub_query_embeddings):
            sub_results = vector_store.similarity_search(embedding, k=2)  # 每个子查询获取较少的结果
            all_results.extend(sub_results)

        # 去重（保留相似度最高的结果）
        seen_texts = {}
        for result in all_results:
            text = result["text"]
            if text not in seen_texts or result["similarity"] > seen_texts[text]["similarity"]:
                seen_texts[text] = result

        # 按相似度排序并取前 top_k 个结果
        results = sorted(seen_texts.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]

    else:
        # 普通搜索（无转换）
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)

    return results


####################################
# 查询重写（Query Rewriting）
####################################
def rewrite_query(original_query):
    """
    重写查询以使其更加具体和详细，从而提高检索效果。

    Args:
        original_query (str): 用户原始查询
        model (str): 用于查询重写的模型

    Returns:
        str: 重写后的查询
    """
    # 定义系统提示，指导AI助手的行为
    system_prompt = "您是一个专注于优化搜索查询的AI助手。您的任务是通过重写用户查询，使其更加具体、详细，并提升检索相关信息的有效性。"

    # 定义用户提示，包含需要重写的原始查询
    user_prompt = f"""
    请优化以下搜索查询，使其满足：
    1. 增强查询的具体性和详细程度
    2. 包含有助于获取准确信息的相关术语和核心概念

    原始查询：{original_query}

    优化后的查询：
    """

    # 使用指定模型生成重写后的查询
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0.0,  # 确保输出的确定性
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回重写后的查询，并去除首尾的空白字符
    return response.choices[0].message.content.strip()


####################################
# 回退提示（Step-back Prompting）
####################################
def generate_step_back_query(original_query):
    """
    生成一个更广泛的“回退”查询以检索更宽泛的上下文信息。

    Args:
        original_query (str): 原始用户查询
        model (str): 用于生成回退查询的模型

    Returns:
        str: 回退查询
    """
    # 定义系统提示，以指导AI助手的行为
    system_prompt = "您是一个专注于搜索策略的AI助手。您的任务是将特定查询转化为更宽泛、更通用的版本，以帮助检索相关背景信息。"

    # 定义用户提示，包含要概括的原始查询
    user_prompt = f"""
    请基于以下具体查询生成更通用的版本，要求：
    1. 扩大查询范围以涵盖背景信息
    2. 包含潜在相关领域的关键概念
    3. 保持语义完整性

    原始查询: {original_query}

    通用化查询：
    """

    # 使用指定的模型生成回退查询
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0.1,  # 稍微高点以增加多样性
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回回退查询，去除任何前导/尾随空格
    return response.choices[0].message.content.strip()


####################################
# 查询分解为子查询（Sub-query Decomposition）
####################################
def decompose_query(original_query, num_subqueries=4):
    """
    将复杂查询分解为更简单的子查询。

    Args:
        original_query (str): 原始的复杂查询
        num_subqueries (int): 要生成的子查询数量
        model (str): 用于查询分解的模型

    Returns:
        List[str]: 更简单子查询的列表
    """
    # 定义系统提示，指导AI助手的行为
    system_prompt = "您是一个专门负责分解复杂问题的AI助手。您的任务是将复杂的查询拆解成更简单的子问题，这些子问题的答案组合起来能够解决原始查询。"

    # 使用需要分解的原始查询定义用户提示
    user_prompt = f"""
    将以下复杂查询分解为{num_subqueries}个更简单的子问题。每个子问题应聚焦原始问题的不同方面。

    原始查询: {original_query}

    请生成{num_subqueries}个子问题，每个问题单独一行，按以下格式：
    1. [第一个子问题]
    2. [第二个子问题]
    依此类推...
    """

    # 使用指定模型生成子查询
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 处理响应以提取子查询
    content = response.choices[0].message.content.strip()

    pattern = r'^\d+\.\s*(.*)'
    return [re.match(pattern, line).group(1) for line in content.split('\n') if line.strip()]

    # 使用简单解析提取编号的查询
    # lines = content.split("\n")
    # sub_queries = []
    # for line in lines:
    #     if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
    #         # 移除编号和前导空格
    #         query = line.strip()
    #         query = query[query.find(".")+1:].strip()
    #         sub_queries.append(query)
    # return sub_queries


def generate_response(query, context):
    """
    根据查询和检索到的上下文生成响应。

    Args:
        query (str): 用户查询
        context (str): 检索到的上下文
    Returns:
        str: 生成的响应
    """
    # 定义系统提示以指导AI助手的行为
    system_prompt = "您是一个乐于助人的AI助手。请仅根据提供的上下文来回答用户的问题。如果在上下文中找不到答案，请直接说'没有足够的信息'。"

    # 定义包含上下文和查询的用户提示
    user_prompt = f"""
        上下文内容:
        {context}

        问题: {query}

        请基于上述上下文内容提供一个全面详尽的答案。
    """

    # 使用指定的模型生成响应
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0,  # 低温度以获得确定性输出
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回生成的响应，去除任何前导/尾随空格
    return response.choices[0].message.content.strip()


def create_embeddings(text):
    """
    使用Embedding模型为给定文本创建嵌入向量。

    Args:
    text (str): 要创建嵌入向量的输入文本。

    Returns:
    List[float]: 嵌入向量。
    """
    # 通过将字符串输入转换为列表来处理字符串和列表输入
    input_text = text if isinstance(text, list) else [text]

    # 使用指定的模型为输入文本创建嵌入向量
    response = client.embeddings.create(
        model=embedding_model,
        input=input_text
    )

    # 如果输入是字符串，仅返回第一个嵌入向量
    if isinstance(text, str):
        return response.data[0].embedding

    # 否则，将所有嵌入向量作为向量列表返回
    return [item.embedding for item in response.data]


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    为RAG处理文档。

    Args:
    pdf_path (str): PDF文件的路径。
    chunk_size (int): 每个文本块的大小（以字符为单位）。
    chunk_overlap (int): 文本块之间的重叠大小（以字符为单位）。

    Returns:
    SimpleVectorStore: 包含文档文本块及其嵌入向量的向量存储。
    """
    print("从PDF中提取文本...")
    extracted_text = extract_text_from_pdf(pdf_path)  # 调用函数提取PDF中的文本

    print("分割文本...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)  # 将提取的文本分割为多个块
    print(f"创建了 {len(chunks)} 个文本块")

    print("为文本块创建嵌入向量...")
    # 为了提高效率，一次性为所有文本块创建嵌入向量
    chunk_embeddings = create_embeddings(chunks)

    # 创建向量存储
    store = SimpleVectorStore()

    # 将文本块添加到向量存储中
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,  # 文本内容
            embedding=embedding,  # 嵌入向量
            metadata={"index": i, "source": pdf_path}  # 元数据，包括索引和源文件路径
        )

    print(f"向向量存储中添加了 {len(chunks)} 个文本块")
    return store


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


def chunk_text(text, n, overlap):
    """
    将文本分割为重叠的块

    Args:
    text (str): 要分割的文本
    n (int): 每个块的字符数
    overlap (int): 块之间的重叠字符数

    Returns:
    List[str]: 文本块列表
    """
    chunks = []  #
    for i in range(0, len(text), n - overlap):
        # 添加从当前索引到索引 + 块大小的文本块
        chunks.append(text[i:i + n])

    return chunks  # Return the list of text chunks


class SimpleVectorStore:
    """
    使用NumPy实现的简单向量存储。
    """

    def __init__(self):
        """
        初始化向量存储。
        """
        self.vectors = []  # 用于存储嵌入向量的列表
        self.texts = []  # 用于存储原始文本的列表
        self.metadata = []  # 用于存储每个文本元数据的列表

    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储中添加一个项目。

        Args:
        text (str): 原始文本。
        embedding (List[float]): 嵌入向量。
        metadata (dict, 可选): 额外的元数据。
        """
        self.vectors.append(np.array(embedding))  # 将嵌入转换为numpy数组并添加到向量列表中
        self.texts.append(text)  # 将原始文本添加到文本列表中
        self.metadata.append(metadata or {})  # 添加元数据到元数据列表中，如果没有提供则使用空字典

    def similarity_search(self, query_embedding, k=5):
        """
        查找与查询嵌入最相似的项目。

        Args:
        query_embedding (List[float]): 查询嵌入向量。
        k (int): 返回的结果数量。

        Returns:
        List[Dict]: 包含文本和元数据的前k个最相似项。
        """
        if not self.vectors:
            return []  # 如果没有存储向量，则返回空列表

        # 将查询嵌入转换为numpy数组
        query_vector = np.array(query_embedding)

        # 使用余弦相似度计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 计算查询向量与存储向量之间的余弦相似度
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 添加索引和相似度分数

        # 按相似度排序（降序）
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # 添加对应的文本
                "metadata": self.metadata[idx],  # 添加对应的元数据
                "similarity": score  # 添加相似度分数
            })

        return results  # 返回前k个最相似项的列表
