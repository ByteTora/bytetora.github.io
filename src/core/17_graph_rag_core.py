"""
Graph RAG 核心函数
"""
import json
import re
import os
import numpy as np
import networkx as nx
import heapq
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
# 从文本块中提取关键概念
####################################
def extract_concepts(text):
    """
    从文本中提取关键概念。

    Args:
        text (str): 需要提取概念的文本

    Returns:
        List[str]: 包含提取出的概念的列表
    """
    # 系统消息，用于指导模型执行任务
    system_message = """从提供的文本中提取关键概念和实体。
只返回一个包含5到10个最重要的关键词、实体或概念的列表
以JSON字符串数组的格式返回结果。

结果格式为：{"concepts": [x, x, x]}

"""

    # 调用OpenAI API进行请求
    response = client.chat.completions.create(
        model=llm_model,  # 指定使用的模型
        messages=[
            {"role": "system", "content": system_message},  # 系统消息
            {"role": "user", "content": f"从以下文本中提取关键概念:\n\n{text[:3000]}"}  # 用户消息，限制文本长度以符合API要求
        ],
        temperature=0.0,  # 设置生成温度为确定性结果
        response_format={"type": "json_object"}  # 指定响应格式为JSON对象
    )

    try:
        # 从响应中解析概念
        concepts_json = json.loads(response.choices[0].message.content.strip())  # 将响应内容解析为JSON
        concepts = concepts_json.get("concepts", [])  # 获取"concepts"字段的值
        if not concepts and "concepts" not in concepts_json:
            # 如果未找到"concepts"字段，则尝试获取JSON中的任意列表
            for key, value in concepts_json.items():
                if isinstance(value, list):
                    concepts = value
                    break
        return concepts  # 返回提取出的概念列表
    except (json.JSONDecodeError, AttributeError):
        # 如果JSON解析失败，则进行回退处理
        content = response.choices[0].message.content  # 获取原始响应内容
        # 尝试从响应内容中提取类似列表的部分
        matches = re.findall(r'\[(.*?)\]', content, re.DOTALL)  # 查找方括号内的内容
        if matches:
            items = re.findall(r'"([^"]*)"', matches[0])  # 提取方括号内的字符串项
            return items
        return []  # 如果无法提取，则返回空列表


####################################
# 构建知识图谱：节点、边
####################################
def build_knowledge_graph(chunks):
    """
    从文本片段构建知识图谱。

    Args:
        chunks (List[Dict]): 包含元数据的文本片段列表

    Returns:
        Tuple[nx.Graph, List[np.ndarray]]: 知识图谱和片段嵌入
    """
    print("正在构建知识图谱...")

    # 创建一个图
    graph = nx.Graph()

    # 提取片段文本
    texts = [chunk["text"] for chunk in chunks]

    # 为所有片段创建嵌入
    print("正在为片段创建嵌入...")
    embeddings = create_embeddings(texts)

    # 将节点添加到图中
    print("正在将节点添加到图中...")
    for i, chunk in enumerate(chunks):
        # 从片段中提取概念
        print(f"正在从片段 {i + 1}/{len(chunks)} 中提取概念...")
        concepts = extract_concepts(chunk["text"])

        # 添加带有属性的节点
        graph.add_node(i,
                       text=chunk["text"],
                       concepts=concepts,
                       embedding=embeddings[i])

    # 根据共享概念连接节点
    print("正在在节点之间创建边...")
    for i in range(len(chunks)):
        node_concepts = set(graph.nodes[i]["concepts"])

        for j in range(i + 1, len(chunks)):
            # 计算概念重叠
            other_concepts = set(graph.nodes[j]["concepts"])
            shared_concepts = node_concepts.intersection(other_concepts)  # 两个节点之间交集

            # 如果它们共享概念，则添加一条边
            if shared_concepts:
                # 使用嵌入计算语义相似性
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))

                # 根据概念重叠和语义相似性计算边权重
                concept_score = len(shared_concepts) / min(len(node_concepts), len(other_concepts))
                edge_weight = 0.7 * similarity + 0.3 * concept_score

                # 仅添加具有显著关系的边
                if edge_weight > 0.6:
                    graph.add_edge(i, j,
                                   weight=edge_weight,
                                   similarity=similarity,
                                   shared_concepts=list(shared_concepts))

    print(f"知识图谱已构建，包含 {graph.number_of_nodes()} 个节点和 {graph.number_of_edges()} 条边")
    return graph, embeddings


####################################
# 遍历知识图谱以查找与查询相关的信息：相似度排序、广度优先搜索
####################################
def traverse_graph(query, graph, embeddings, top_k=5, max_depth=3):
    """
    遍历知识图谱以查找与查询相关的信息。

    Args:
        query (str): 用户的问题
        graph (nx.Graph): 知识图谱
        embeddings (List): 节点嵌入列表
        top_k (int): 考虑的初始节点数量
        max_depth (int): 最大遍历深度

    Returns:
        List[Dict]: 图遍历得到的相关信息
    """
    print(f"正在为查询遍历图: {query}")

    # 获取查询的嵌入
    query_embedding = create_embeddings(query)

    # 计算查询与所有节点之间的相似度
    similarities = []
    for i, node_embedding in enumerate(embeddings):
        similarity = np.dot(query_embedding, node_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
        similarities.append((i, similarity))

    # 按相似度排序（降序）
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 获取最相似的前 top-k 个节点作为起点
    starting_nodes = [node for node, _ in similarities[:top_k]]
    print(f"从 {len(starting_nodes)} 个节点开始遍历")

    # 初始化遍历
    visited = set()  # 用于跟踪已访问节点的集合
    traversal_path = []  # 存储遍历路径的列表
    results = []  # 存储结果的列表

    # 使用优先队列进行遍历
    queue = []
    for node in starting_nodes:
        heapq.heappush(queue, (-similarities[node][1], node))  # 负号用于最大堆

    # 使用修改后的基于优先级的广度优先搜索遍历图
    while queue and len(results) < (top_k * 3):  # 将结果限制为 top_k * 3
        _, node = heapq.heappop(queue)

        if node in visited:
            continue

        # 标记为已访问
        visited.add(node)
        traversal_path.append(node)

        # 将当前节点的文本添加到结果中
        results.append({
            "text": graph.nodes[node]["text"],
            "concepts": graph.nodes[node]["concepts"],
            "node_id": node
        })

        # 如果尚未达到最大深度，则探索邻居
        if len(traversal_path) < max_depth:
            neighbors = [(neighbor, graph[node][neighbor]["weight"])
                         for neighbor in graph.neighbors(node)
                         if neighbor not in visited]

            # 根据边权重将邻居添加到队列中
            for neighbor, weight in sorted(neighbors, key=lambda x: x[1], reverse=True):
                heapq.heappush(queue, (-weight, neighbor))

    print(f"图遍历找到了 {len(results)} 个相关片段")
    return results, traversal_path


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
