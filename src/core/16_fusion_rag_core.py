"""
混合检索核心函数
"""
import os
import numpy as np
import jieba
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
# 混合检索
####################################
def fusion_retrieval(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    """
    执行融合检索，结合基于向量和BM25的搜索。

    Args:
        query (str): 查询字符串
        chunks (List[Dict]): 原始文本块
        vector_store (SimpleVectorStore): 向量存储
        bm25_index (BM25Okapi): BM25 索引
        k (int): 返回的结果数量
        alpha (float): 向量分数的权重（0-1），其中 1-alpha 是 BM25 的权重

    Returns:
        List[Dict]: 基于综合分数的前 k 个结果
    """
    print(f"正在为查询执行融合检索: {query}")

    # 定义一个小的 epsilon 来避免除以零
    epsilon = 1e-8

    # 获取向量搜索结果
    query_embedding = create_embeddings(query)  # 为查询创建嵌入
    vector_results = vector_store.similarity_search_with_scores(query_embedding, k=len(chunks))  # 执行向量搜索

    # 获取 BM25 搜索结果
    bm25_results = bm25_search(bm25_index, chunks, query, k=len(chunks))  # 执行 BM25 搜索

    # 创建字典将文档索引映射到分数
    vector_scores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}

    # 确保所有文档都有两种方法的分数
    all_docs = vector_store.get_all_documents()
    combined_results = []

    for i, doc in enumerate(all_docs):
        vector_score = vector_scores_dict.get(i, 0.0)  # 获取向量分数，如果未找到则为 0
        bm25_score = bm25_scores_dict.get(i, 0.0)  # 获取 BM25 分数，如果未找到则为 0
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })

    # 提取分数为数组
    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])

    # 归一化分数
    norm_vector_scores = (vector_scores - np.min(vector_scores)) / (
                np.max(vector_scores) - np.min(vector_scores) + epsilon)
    norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

    # 计算综合分数
    combined_scores = alpha * norm_vector_scores + (1 - alpha) * norm_bm25_scores

    # 将综合分数添加到结果中
    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)

    # 按综合分数排序（降序）
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

    # 返回前 k 个结果
    top_results = combined_results[:k]

    print(f"通过融合检索获取了 {len(top_results)} 份文档")
    return top_results


####################################
# BM25 检索
####################################
def bm25_search(bm25, chunks, query, k=5):
    """
    使用查询在 BM25 索引中进行搜索。

    Args:
        bm25 (BM25Okapi): BM25 索引
        chunks (List[Dict]): 文本块列表
        query (str): 查询字符串
        k (int): 返回的结果数量

    Returns:
        List[Dict]: 带有分数的前 k 个结果
    """
    # 将查询按空格分割成单独的词
    # query_tokens = query.split()  # 英文
    query_tokens = list(jieba.cut(query))  # 中文

    # 获取查询词对已索引文档的 BM25 分数
    scores = bm25.get_scores(query_tokens)

    # 初始化一个空列表，用于存储带有分数的结果
    results = []

    # 遍历分数和对应的文本块
    for i, score in enumerate(scores):
        # 创建元数据的副本以避免修改原始数据
        metadata = chunks[i].get("metadata", {}).copy()
        # 向元数据中添加索引
        metadata["index"] = i

        results.append({
            "text": chunks[i]["text"],  # 文本内容
            "metadata": metadata,  # 带索引的元数据
            "bm25_score": float(score)  # BM25 分数
        })

    # 按 BM25 分数降序排序结果
    results.sort(key=lambda x: x["bm25_score"], reverse=True)

    # 返回前 k 个结果
    return results[:k]


def create_embeddings(texts):
    """
    为给定的文本创建嵌入向量。

    Args:
        texts (str 或 List[str]): 输入文本（可以是单个字符串或字符串列表）
        # model (str): 嵌入模型名称

    返回:
        List[List[float]]: 嵌入向量列表
    """
    # 处理字符串和列表类型的输入
    input_texts = texts if isinstance(texts, list) else [texts]

    # 如果需要，按批次处理（OpenAI API 有请求限制）
    batch_size = 100
    all_embeddings = []

    # 按批次迭代输入文本
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]  # 获取当前批次的文本

        # 为当前批次创建嵌入向量
        response = client.embeddings.create(
            model=embedding_model,
            input=batch
        )

        # 从响应中提取嵌入向量
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # 将批次嵌入向量添加到总列表中

    # 如果输入是单个字符串，仅返回第一个嵌入向量
    if isinstance(texts, str):
        return all_embeddings[0]

    # 否则，返回所有嵌入向量
    return all_embeddings
