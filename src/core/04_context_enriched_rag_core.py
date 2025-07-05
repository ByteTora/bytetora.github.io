"""
上下文丰富检索 核心函数
"""
import os
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
# 检索最相关文本块以及相邻上下文
####################################
def context_enriched_search(query, text_chunks, embeddings, k=1, context_size=1):
    """
    检索最相关的文本块及其相邻的上下文块

    Args:
        query (str): 搜索查询
        text_chunks (List[str]): 文本块列表
        embeddings (List[dict]): 文本块嵌入列表
        k (int): 要检索的相关块数量
        context_size (int): 包含的相邻块数量

    Returns:
        List[str]: 包含上下文信息的相关文本块
    """
    # 将查询转换为嵌入向量
    query_embedding = create_embeddings(query)
    similarity_scores = []

    # 计算查询与每个文本块嵌入之间的相似度分数
    for i, chunk_embedding in enumerate(embeddings):
        # 计算查询嵌入与当前文本块嵌入之间的余弦相似度
        similarity_score = cosine_similarity(np.array(query_embedding), chunk_embedding)
        # 将索引和相似度分数存储为元组
        similarity_scores.append((i, similarity_score))

    # 按相似度分数降序排序（相似度最高排在前面）
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 获取最相关块的索引
    # top_index = [index for index, _ in similarity_scores[:k]]
    top_index = similarity_scores[0][0]

    # similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    # top_indices = np.argsort(similarities)[-k:][::-1]

    # 定义上下文包含的范围
    # 确保不会超出 text_chunks 的边界
    start = max(0, top_index - context_size)
    end = min(len(text_chunks), top_index + context_size + 1)

    # 返回最相关的块及其相邻的上下文块
    return [text_chunks[i] for i in range(start, end)]


def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity score.
    """

    # Compute the dot product of the two vectors
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def create_embeddings(texts):
    """
    为文本列表生成嵌入

    Args:
        texts (List[str]): 输入文本列表.

    Returns:
        List[np.ndarray]: List of numerical embeddings.
    """
    # 确保每次调用不超过64条文本
    batch_size = 64
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL_ID"),
            input=batch
        )
        # 将响应转换为numpy数组列表并添加到embeddings列表中
        embeddings.extend([np.array(embedding.embedding) for embedding in response.data])

    return embeddings
