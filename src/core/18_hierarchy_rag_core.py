"""
分级检索 核心函数
"""
import os
import pickle
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
# 分级索引 RAG 流程
####################################
def hierarchical_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200,
                     k_summaries=3, k_chunks=5, regenerate=False):
    """
    完整的分级 RAG 管道。

    Args:
        query (str): 用户查询
        pdf_path (str): PDF 文档的路径
        chunk_size (int): 每个详细块的大小
        chunk_overlap (int): 块之间的重叠
        k_summaries (int): 要检索的摘要数量
        k_chunks (int): 每个摘要要检索的块数量
        regenerate (bool): 是否重新生成向量存储

    Returns:
        Dict: 包括响应和检索到的块的结果
    """
    # 创建用于缓存的存储文件名
    summary_store_file = f"{os.path.basename(pdf_path)}_summary_store.pkl"
    detailed_store_file = f"{os.path.basename(pdf_path)}_detailed_store.pkl"

    # 如果需要，处理文档并创建存储
    if regenerate or not os.path.exists(summary_store_file) or not os.path.exists(detailed_store_file):
        print("处理文档并创建向量存储...")
        # 处理文档以创建分层索引和向量存储
        summary_store, detailed_store = process_document_hierarchically(
            pdf_path, chunk_size, chunk_overlap
        )

        # 将摘要存储保存到文件以供将来使用
        with open(summary_store_file, 'wb') as f:
            pickle.dump(summary_store, f)

        # 将详细存储保存到文件以供将来使用
        with open(detailed_store_file, 'wb') as f:
            pickle.dump(detailed_store, f)
    else:
        # 从文件加载现有的摘要存储
        print("加载现有的向量存储...")
        with open(summary_store_file, 'rb') as f:
            summary_store = pickle.load(f)

        # 从文件加载现有的详细存储
        with open(detailed_store_file, 'rb') as f:
            detailed_store = pickle.load(f)

    # 使用查询分层检索相关块
    retrieved_chunks = retrieve_hierarchically(
        query, summary_store, detailed_store, k_summaries, k_chunks
    )

    # 根据检索到的块生成响应
    response = generate_response(query, retrieved_chunks)

    # 返回结果，包括查询、响应、检索到的块以及摘要和详细块的数量
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks,
        "summary_count": len(summary_store.texts),
        "detailed_count": len(detailed_store.texts)
    }


####################################
# 将文档处理为分级索引：页面摘要、详细块
####################################
def process_document_hierarchically(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    将文档处理为分层索引。

    Args:
        pdf_path (str): PDF 文件的路径
        chunk_size (int): 每个详细块的大小
        chunk_overlap (int): 块之间的重叠量

    Returns:
        Tuple[SimpleVectorStore, SimpleVectorStore]: 摘要和详细向量存储
    """
    # 从 PDF 中提取页面
    pages = extract_text_from_pdf(pdf_path)

    # 为每一页创建摘要
    print("生成页面摘要...")
    summaries = []
    for i, page in enumerate(pages):
        print(f"正在摘要第 {i + 1}/{len(pages)} 页...")
        summary_text = generate_page_summary(page["text"])

        # 创建摘要元数据
        summary_metadata = page["metadata"].copy()
        summary_metadata.update({"is_summary": True})

        # 将摘要文本和元数据添加到摘要列表中
        summaries.append({
            "text": summary_text,
            "metadata": summary_metadata
        })

    # 为每一页创建详细块
    detailed_chunks = []
    for page in pages:
        # 将页面的文本切分为块
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        # 使用当前页面的块扩展 detailed_chunks 列表
        detailed_chunks.extend(page_chunks)

    print(f"已创建 {len(detailed_chunks)} 个详细块")

    # 为摘要创建嵌入
    print("正在为摘要创建嵌入...")
    summary_texts = [summary["text"] for summary in summaries]
    summary_embeddings = create_embeddings(summary_texts)

    # 为详细块创建嵌入
    print("正在为详细块创建嵌入...")
    chunk_texts = [chunk["text"] for chunk in detailed_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)

    # 创建向量存储
    summary_store = SimpleVectorStore()
    detailed_store = SimpleVectorStore()

    # 将摘要添加到摘要存储中
    for i, summary in enumerate(summaries):
        summary_store.add_item(
            text=summary["text"],
            embedding=summary_embeddings[i],
            metadata=summary["metadata"]
        )

    # 将块添加到详细存储中
    for i, chunk in enumerate(detailed_chunks):
        detailed_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )

    print(f"已创建包含 {len(summaries)} 个摘要和 {len(detailed_chunks)} 个块的向量存储")
    return summary_store, detailed_store


####################################
# 分级索引检索信息：先搜索摘要，再搜索块
####################################
def retrieve_hierarchically(query, summary_store, detailed_store, k_summaries=3, k_chunks=5):
    """
    使用分级索引检索信息。

    Args:
        query (str): 用户查询
        summary_store (SimpleVectorStore): 文档摘要存储
        detailed_store (SimpleVectorStore): 详细块存储
        k_summaries (int): 要检索的摘要数量
        k_chunks (int): 每个摘要要检索的块数量

    Returns:
        List[Dict]: 检索到的带有相关性分数的块
    """
    print(f"正在为查询执行分层检索: {query}")

    # 创建查询嵌入
    query_embedding = create_embeddings(query)

    # 首先，检索相关的摘要
    summary_results = summary_store.similarity_search(
        query_embedding,
        k=k_summaries
    )

    print(f"检索到 {len(summary_results)} 个相关摘要")

    # 收集来自相关摘要的页面
    relevant_pages = [result["metadata"]["page"] for result in summary_results]

    # 创建一个过滤函数，仅保留来自相关页面的块
    def page_filter(metadata):
        return metadata["page"] in relevant_pages

    # 然后，仅从这些相关页面检索详细块
    detailed_results = detailed_store.similarity_search(
        query_embedding,
        k=k_chunks * len(relevant_pages),
        filter_func=page_filter
    )

    print(f"从相关页面检索到 {len(detailed_results)} 个详细块")

    # 对于每个结果，添加它来自哪个摘要/页面
    for result in detailed_results:
        page = result["metadata"]["page"]
        matching_summaries = [s for s in summary_results if s["metadata"]["page"] == page]
        if matching_summaries:
            result["summary"] = matching_summaries[0]["text"]

    return detailed_results


####################################
# 分级索引检索信息：先搜索摘要，再搜索块
####################################
def generate_page_summary(page_text):
    """
    生成页面的简洁摘要。

    Args:
        page_text (str): 页面的文本内容

    Returns:
        str: 生成的摘要
    """
    # 定义系统提示，指导摘要模型如何生成摘要
    system_prompt = """你是一个专业的摘要生成系统。
    请对提供的文本创建一个详细的摘要。
    重点捕捉主要内容、关键信息和重要事实。
    你的摘要应足够全面，能够让人理解该页面包含的内容，
    但要比原文更简洁。"""

    # 如果输入文本超过最大令牌限制，则截断
    max_tokens = 6000
    truncated_text = page_text[:max_tokens] if len(page_text) > max_tokens else page_text

    # 向OpenAI API发出请求以生成摘要
    response = client.chat.completions.create(
        model=llm_model,  # 指定要使用的模型
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息以引导助手
            {"role": "user", "content": f"请总结以下文本:\n\n{truncated_text}"}  # 用户消息，包含要总结的文本
        ],
        temperature=0.3  # 设置响应生成的温度
    )

    # 返回生成的摘要内容
    return response.choices[0].message.content


def create_embeddings(texts):
    """
    为给定文本创建嵌入向量。

    Args:
        texts (List[str]): 输入文本列表
        model (str): 嵌入模型名称

    Returns:
        List[List[float]]: 嵌入向量列表
    """
    # 处理空输入的情况
    if not texts:
        return []

    # 分批次处理（OpenAI API 的限制）
    batch_size = 100
    all_embeddings = []

    # 遍历输入文本，按批次生成嵌入
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # 获取当前批次的文本

        # 调用 OpenAI 接口生成嵌入
        response = client.embeddings.create(
            model=embedding_model,
            input=batch
        )

        # 提取当前批次的嵌入向量
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # 将当前批次的嵌入向量加入总列表

    return all_embeddings  # 返回所有嵌入向量


def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容，并按页分离。

    Args:
        pdf_path (str): PDF文件的路径

    Returns:
        List[Dict]: 包含文本内容和元数据的页面列表
    """
    print(f"正在提取文本 {pdf_path}...")  # 打印正在处理的PDF路径
    pdf = fitz.open(pdf_path)  # 使用PyMuPDF打开PDF文件
    pages = []  # 初始化一个空列表，用于存储包含文本内容的页面

    # 遍历PDF中的每一页
    for page_num in range(len(pdf)):
        page = pdf[page_num]  # 获取当前页
        text = page.get_text()  # 从当前页提取文本

        # 跳过文本非常少的页面（少于50个字符）
        if len(text.strip()) > 50:
            # 将页面文本和元数据添加到列表中
            pages.append({
                "text": text,
                "metadata": {
                    "source": pdf_path,  # 源文件路径
                    "page": page_num + 1  # 页面编号（从1开始）
                }
            })

    print(f"已提取 {len(pages)} 页的内容")  # 打印已提取的页面数量
    return pages  # 返回包含文本内容和元数据的页面列表


def chunk_text(text, metadata, chunk_size=1000, overlap=200):
    """
    将文本分割为重叠的块，同时保留元数据。

    Args:
        text (str): 要分割的输入文本
        metadata (Dict): 要保留的元数据
        chunk_size (int): 每个块的大小（以字符为单位）
        overlap (int): 块之间的重叠大小（以字符为单位）

    Returns:
        List[Dict]: 包含元数据的文本块列表
    """
    chunks = []  # 初始化一个空列表，用于存储块

    # 按指定的块大小和重叠量遍历文本
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]  # 提取文本块

        # 跳过非常小的块（少于50个字符）
        if chunk_text and len(chunk_text.strip()) > 50:
            # 创建元数据的副本，并添加块特定的信息
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),  # 块的索引
                "start_char": i,  # 块的起始字符索引
                "end_char": i + len(chunk_text),  # 块的结束字符索引
                "is_summary": False  # 标志，表示这不是摘要
            })

            # 将带有元数据的块添加到列表中
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

    return chunks  # 返回带有元数据的块列表


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
            metadata (dict, optional): 额外的元数据。
        """
        self.vectors.append(np.array(embedding))  # 将嵌入转换为numpy数组并添加到向量列表中
        self.texts.append(text)  # 将原始文本添加到文本列表中
        self.metadata.append(metadata or {})  # 添加元数据到元数据列表中，如果没有提供则使用空字典

    def similarity_search(self, query_embedding, k=5, filter_func=None):
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
            # 如果存在过滤函数且该元数据不符合条件，则跳过该项
            if filter_func and not filter_func(self.metadata[i]):
                continue
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


def generate_response(query, retrieved_chunks):
    """
    根据查询和检索到的块生成响应。

    Args:
        query (str): 用户查询
        retrieved_chunks (List[Dict]): 从分层搜索中检索到的块

    Returns:
        str: 生成的响应
    """
    # 从块中提取文本并准备上下文部分
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks):
        page_num = chunk["metadata"]["page"]  # 从元数据中获取页码
        context_parts.append(f"[Page {page_num}]: {chunk['text']}")  # 使用页码格式化块文本

    # 将所有上下文部分合并为一个上下文字符串
    context = "\n\n".join(context_parts)

    # 定义系统消息以指导AI助手
    system_message = """你是一个乐于助人的AI助手，根据提供的上下文回答问题。
请准确利用上下文中的信息来回答用户的问题。
如果上下文中不包含相关信息，请予以说明。
引用具体信息时请注明页码。"""

    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model=llm_model,  # 指定要使用的模型
        messages=[
            {"role": "system", "content": system_message},  # 系统消息以指导助手
            {"role": "user", "content": f"上下文内容:\n\n{context}\n\n查询问题: {query}"}  # 包含上下文和查询的用户消息
        ],
        temperature=0.2  # 设置用于响应生成的温度
    )

    # 返回生成的响应内容
    return response.choices[0].message.content
