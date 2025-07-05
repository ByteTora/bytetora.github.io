"""
语义块切分 核心函数
"""
import numpy as np

####################################
# 根据相似度下降计算分块的断点:断点方法有三种：百分位、标准差和四分位距
####################################
def compute_breakpoints(similarities, method="percentile", threshold=90):
    """
    根据相似度下降计算分块的断点。

    Args:
        similarities (List[float]): 句子之间的相似度分数列表。
        method (str): 'percentile'（百分位）、'standard_deviation'（标准差）或 'interquartile'（四分位距）。
        threshold (float): 阈值（对于 'percentile' 是百分位数，对于 'standard_deviation' 是标准差倍数）。

    Returns:
        List[int]: 分块的索引列表。
    """
    # 根据选定的方法确定阈值
    if method == "percentile":
        # 计算相似度分数的第 X 百分位数
        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        # 计算相似度分数的均值和标准差。
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        # 将阈值设置为均值减去 X 倍的标准差
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        # 计算第一和第三四分位数（Q1 和 Q3）。
        q1, q3 = np.percentile(similarities, [25, 75])
        # 使用 IQR 规则（四分位距规则）设置阈值
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        # 如果提供了无效的方法，则抛出异常
        raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

    # 找出相似度低于阈值的索引
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]

# # 使用百分位法计算断点，阈值为90
# breakpoints = compute_breakpoints(similarities, method="percentile", threshold=90)
# breakpoints

####################################
# 根据断点分割文本，得到语义块
####################################
def split_into_chunks(sentences, breakpoints):
    """
    将句子分割为语义块

    Args:
    sentences (List[str]): 句子列表
    breakpoints (List[int]): 进行分块的索引位置

    Returns:
    List[str]: 文本块列表
    """
    chunks = []  # Initialize an empty list to store the chunks
    start = 0  # Initialize the start index

    # 遍历每个断点以创建块
    for bp in breakpoints:
        # 将从起始位置到当前断点的句子块追加到列表中
        chunks.append("。".join(sentences[start:bp + 1]) + "。")
        start = bp + 1  # 将起始索引更新为断点后的下一个句子

    # 将剩余的句子作为最后一个块追加
    chunks.append("。".join(sentences[start:]))
    return chunks  # Return the list of chunks

# # split_into_chunks 函数创建文本块
# text_chunks = split_into_chunks(sentences, breakpoints)
#
# # Print the number of chunks created
# print(f"Number of semantic chunks: {len(text_chunks)}")
#
# # Print the first chunk to verify the result
# print("\nFirst text chunk:")
# print(text_chunks[0])
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): First vector.
    vec2 (np.ndarray): Second vector.

    Returns:
    float: Cosine similarity.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compute similarity between consecutive sentences
# similarities = [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]









