"""
上下文情境压缩（Contextual Compression） 核心函数
"""
import os
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
# 上下文压缩RAG完整流程
####################################
def rag_with_compression(pdf_path, query, k=10, compression_type="selective"):
    """
    完整的RAG管道，包含上下文压缩。

    Args:
        pdf_path (str): PDF文档的路径
        query (str): 用户查询
        k (int): 初始检索的块数量
        compression_type (str): 压缩类型

    Returns:
        dict: 包括查询、压缩块和响应的结果
    """
    print("\n=== RAG WITH CONTEXTUAL COMPRESSION ===")
    print(f"Query: {query}")
    print(f"Compression type: {compression_type}")

    # 处理文档以提取文本、分块并创建嵌入
    vector_store = process_document(pdf_path)

    # 为查询创建嵌入
    query_embedding = create_embeddings(query)

    # 根据查询嵌入检索最相似的前k个块
    print(f"Retrieving top {k} chunks...")
    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]

    # 对检索到的块应用压缩
    compressed_results = batch_compress_chunks(retrieved_chunks, query, compression_type)
    compressed_chunks = [result[0] for result in compressed_results]
    compression_ratios = [result[1] for result in compressed_results]

    # 过滤掉任何空的压缩块
    filtered_chunks = [(chunk, ratio) for chunk, ratio in zip(compressed_chunks, compression_ratios) if chunk.strip()]

    if not filtered_chunks:
        # 如果所有块都被压缩为空字符串，则使用原始块
        print("Warning: All chunks were compressed to empty strings. Using original chunks.")
        filtered_chunks = [(chunk, 0.0) for chunk in retrieved_chunks]
    else:
        compressed_chunks, compression_ratios = zip(*filtered_chunks)

    # 从压缩块生成上下文
    context = "\n\n---\n\n".join(compressed_chunks)

    # 基于压缩块生成响应
    print("Generating response based on compressed chunks...")
    response = generate_response(query, context)

    # 准备结果字典
    result = {
        "query": query,
        "original_chunks": retrieved_chunks,
        "compressed_chunks": compressed_chunks,
        "compression_ratios": compression_ratios,
        "context_length_reduction": f"{sum(compression_ratios) / len(compression_ratios):.2f}%",
        "response": response
    }

    print("\n=== RESPONSE ===")
    print(response)

    return result


####################################
# 压缩检索到的文本块：selective、summary、extraction
####################################
def compress_chunk(chunk, query, compression_type="selective"):
    """
    压缩检索到的文本块，仅保留与查询相关的内容。

    Args:
        chunk (str): 要压缩的文本块
        query (str): 用户查询
        compression_type (str): 压缩类型 ("selective", "summary" 或 "extraction")

    Returns:
        str: 压缩后的文本块
    """
    # 为不同的压缩方法定义系统提示
    if compression_type == "selective":
        system_prompt = """您是专业信息过滤专家。
        您的任务是分析文档块并仅提取与用户查询直接相关的句子或段落，移除所有无关内容。

        输出要求：
        1. 仅保留有助于回答查询的文本
        2. 保持相关句子的原始措辞（禁止改写）
        3. 维持文本的原始顺序
        4. 包含所有相关文本（即使存在重复）
        5. 排除任何与查询无关的文本

        请以纯文本格式输出，不添加任何注释。"""

    elif compression_type == "summary":
        system_prompt = """您是专业摘要生成专家。
        您的任务是创建文档块的简洁摘要，且仅聚焦与用户查询相关的信息。

        输出要求：
        1. 保持简明扼要但涵盖所有相关要素
        2. 仅聚焦与查询直接相关的信息
        3. 省略无关细节
        4. 使用中立、客观的陈述语气

        请以纯文本格式输出，不添加任何注释。"""

    else:  # extraction
        system_prompt = """您是精准信息提取专家。
        您的任务是从文档块中精确提取与用户查询相关的完整句子。

        输出要求：
        1. 仅包含原始文本中的直接引用
        2. 严格保持原始文本的措辞（禁止修改）
        3. 仅选择与查询直接相关的完整句子
        4. 不同句子使用换行符分隔
        5. 不添加任何解释性文字

        请以纯文本格式输出，不添加任何注释。"""

    # 定义带有查询和文档块的用户提示
    user_prompt = f"""
        查询: {query}

        文档块:
        {chunk}

        请严格提取与本查询相关的核心内容。
    """

    # 使用 OpenAI API 生成响应
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从响应中提取压缩后的文本块
    compressed_chunk = response.choices[0].message.content.strip()

    # 计算压缩比率
    original_length = len(chunk)
    compressed_length = len(compressed_chunk)
    compression_ratio = (original_length - compressed_length) / original_length * 100

    return compressed_chunk, compression_ratio


####################################
# 批量压缩检索到的文本块
####################################
def batch_compress_chunks(chunks, query, compression_type="selective"):
    """
    逐个压缩多个文本块。

    Args:
        chunks (List[str]): 要压缩的文本块列表
        query (str): 用户查询
        compression_type (str): 压缩类型 ("selective", "summary", 或 "extraction")

    Returns:
        List[Tuple[str, float]]: 包含压缩比率的压缩文本块列表
    """
    print(f"正在压缩 {len(chunks)} 个文本块...")  # 打印将要压缩的文本块数量
    results = []  # 初始化一个空列表以存储结果
    total_original_length = 0  # 初始化变量以存储所有文本块的原始总长度
    total_compressed_length = 0  # 初始化变量以存储所有文本块的压缩后总长度

    # 遍历每个文本块
    for i, chunk in enumerate(chunks):
        print(f"正在压缩文本块 {i + 1}/{len(chunks)}...")  # 打印压缩进度
        # 压缩文本块并获取压缩后的文本块和压缩比率
        compressed_chunk, compression_ratio = compress_chunk(chunk, query, compression_type)
        results.append((compressed_chunk, compression_ratio))  # 将结果添加到结果列表中

        total_original_length += len(chunk)  # 将原始文本块的长度加到总原始长度中
        total_compressed_length += len(compressed_chunk)  # 将压缩后文本块的长度加到总压缩长度中

    # 计算总体压缩比率
    overall_ratio = (total_original_length - total_compressed_length) / total_original_length * 100
    print(f"总体压缩比率: {overall_ratio:.2f}%")  # 打印总体压缩比率

    return results  # 返回包含压缩文本块和压缩比率的列表


def generate_response(query, context):
    """
    根据查询和上下文生成响应。

    Args:
        query (str): 用户查询
        context (str): 从压缩块中提取的上下文文本

    Returns:
        str: 生成的响应
    """
    # 定义系统提示以指导AI的行为
    system_prompt = "您是一个乐于助人的AI助手。请仅根据提供的上下文来回答用户的问题。如果在上下文中找不到答案，请直接说'没有足够的信息'。"

    # 通过组合上下文和查询创建用户提示
    user_prompt = f"""
        上下文:
        {context}

        问题: {query}

        请基于上述上下文内容提供一个全面详尽的答案。
    """

    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 返回生成的响应内容
    return response.choices[0].message.content


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
        chunk = text[i:i + n]
        if chunk:
            chunks.append(chunk)

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
