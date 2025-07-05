# 纠错型RAG（CRAG）实现

实现**纠错型RAG（Corrective RAG）**——一种先进的方法，能够动态评估检索到的信息，并在必要时对检索过程进行修正，使用网络搜索作为备选方案。

-----
CRAG 在传统 RAG 的基础上进行了以下改进：

- 在使用前对检索到的内容进行评估
- 根据相关性动态切换不同的知识源
- 当本地知识不足以回答问题时，通过网络搜索修正检索结果
- 在适当时合并多个来源的信息

-----
实现步骤：
- 处理文档并创建向量数据库
- 创建查询嵌入并检索文档
- 评估文档相关性：对检索到的内容进行评估。
- 根据情况执行相应的知识获取策略：高相关性（评估分数>0.7）,直接使用文档内容；低相关性（评估分数<0.3）使用网络搜索；中等相关性（0.3-0.7）结合文档与网络搜索结果，并将文档结果与网络搜索结果进行合并。在混合搜索中，需要将搜索出来的内容，进行模型提炼，避免内容重复冗余。
- 生成最终回答


```python
import fitz
import os
import re
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from typing import List, Dict, Tuple, Any
import pickle
import requests
from urllib.parse import quote_plus

load_dotenv()
```




    True




```python
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
llm_model = os.getenv("LLM_MODEL_ID")
embedding_model = os.getenv("EMBEDDING_MODEL_ID")

pdf_path = "../../data/AI_Information.en.zh-CN.pdf"
```

## 文档处理函数


```python
def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容。

    Args:
        pdf_path (str): PDF文件的路径

    Returns:
        str: 提取出的文本内容
    """
    print(f"正在从 {pdf_path} 提取文本...")

    # 打开PDF文件
    pdf = fitz.open(pdf_path)
    text = ""

    # 遍历PDF中的每一页
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        # 从当前页提取文本并追加到text变量中
        text += page.get_text()

    return text
```


```python
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    将文本分割为有重叠的块，以便进行高效检索和处理。

    该函数将大段文本划分为较小且易于管理的文本块，并在连续块之间设置指定的重叠字符数。
    对于RAG系统来说，分块非常关键，因为它可以实现更精确的相关信息检索。

    Args:
        text (str): 要分块的输入文本
        chunk_size (int): 每个块的最大字符数
        overlap (int): 连续块之间的重叠字符数，用于保持跨块边界的上下文连贯性

    Returns:
        List[Dict]: 文本块列表，每个块包含：
                   - text: 块内容
                   - metadata: 包含位置信息和来源类型的字典
    """
    chunks = []

    # 使用滑动窗口方式遍历文本
    # 每次移动 (chunk_size - overlap) 的距离以确保块之间有适当重叠
    for i in range(0, len(text), chunk_size - overlap):
        # 提取当前块的内容，不超过chunk_size
        chunk_text = text[i:i + chunk_size]

        # 仅添加非空的文本块
        if chunk_text:
            chunks.append({
                "text": chunk_text,  # 实际的文本内容
                "metadata": {
                    "start_pos": i,  # 在原文本中的起始位置
                    "end_pos": i + len(chunk_text),  # 结束位置
                    "source_type": "document"  # 表示此文本的来源类型
                }
            })

    print(f"共创建了 {len(chunks)} 个文本块")
    return chunks
```

## 向量存储


```python
class SimpleVectorStore:
    """
    一个使用 NumPy 实现的简单向量存储。
    """
    def __init__(self):
        # 初始化列表用于存储向量、文本和元数据
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        """
        向向量库中添加一项数据。

        Args:
            text (str): 文本内容
            embedding (List[float]): 嵌入向量
            metadata (Dict, optional): 额外的元数据
        """
        # 将嵌入向量、文本和元数据分别加入对应的列表中
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def add_items(self, items, embeddings):
        """
        批量添加多个项到向量库中。

        Args:
            items (List[Dict]): 包含文本和元数据的项列表
            embeddings (List[List[float]]): 嵌入向量列表
        """
        # 遍历items和embeddings，逐个添加至向量库
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"],
                embedding=embedding,
                metadata=item.get("metadata", {})
            )

    def similarity_search(self, query_embedding, k=5):
        """
        查找与查询嵌入最相似的k个条目。

        Args:
            query_embedding (List[float]): 查询嵌入向量
            k (int): 返回结果的数量

        Returns:
            List[Dict]: 最相似的前k个条目，包含文本、元数据和相似度分数
        """
        # 如果向量库为空，则返回空列表
        if not self.vectors:
            return []

        # 将查询向量转换为numpy数组
        query_vector = np.array(query_embedding)

        # 计算相似度（使用余弦相似度）
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append((i, similarity))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })

        return results
```

## 创建嵌入


```python
def create_embeddings(texts):
    """
    为文本输入创建向量嵌入。

    嵌入是文本的密集向量表示，能够捕捉语义含义，便于进行相似性比较。
    在 RAG 系统中，嵌入对于将查询与相关文档块进行匹配非常关键。

    Args:
        texts (str 或 List[str]): 要嵌入的输入文本。可以是单个字符串或字符串列表。
        model (str): 要使用的嵌入模型名称。默认为 "text-embedding-3-small"。

    Returns:
        List[List[float]]: 如果输入是列表，返回每个文本对应的嵌入向量列表；
                          如果输入是单个字符串，返回一个嵌入向量。
    """
    # 处理单个字符串和列表两种输入形式：统一转为列表处理
    input_texts = texts if isinstance(texts, list) else [texts]

    # 分批次处理以避免 API 速率限制和请求体大小限制
    batch_size = 100
    all_embeddings = []

    # 遍历每一批文本
    for i in range(0, len(input_texts), batch_size):
        # 提取当前批次的文本
        batch = input_texts[i:i + batch_size]

        # 调用 API 生成当前批次的嵌入
        response = client.embeddings.create(
            model=embedding_model,
            input=batch
        )

        # 从响应中提取嵌入向量并加入总结果中
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    # 如果原始输入是单个字符串，则只返回第一个嵌入
    if isinstance(texts, str):
        return all_embeddings[0]

    # 否则返回所有嵌入组成的列表
    return all_embeddings
```

## 文档处理流程


```python
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    将文档处理并存入向量库中。

    Args:
        pdf_path (str): PDF 文件的路径
        chunk_size (int): 每个文本块的字符数
        chunk_overlap (int): 文本块之间的重叠字符数

    Returns:
        SimpleVectorStore: 包含文档块及其嵌入的向量库
    """
    # 从PDF文件中提取文本
    text = extract_text_from_pdf(pdf_path)

    # 将提取到的文本按指定大小和重叠度进行分块
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # 为每个文本块生成嵌入向量
    print("正在为文本块生成嵌入...")
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = create_embeddings(chunk_texts)

    # 初始化一个新的向量存储
    vector_store = SimpleVectorStore()

    # 将文本块及其嵌入添加到向量库中
    vector_store.add_items(chunks, chunk_embeddings)

    print(f"已创建包含 {len(chunks)} 个文本块的向量库")
    return vector_store
```

## 相关性评价函数



```python
def evaluate_document_relevance(query, document):
    """
    评估文档与查询的相关性。

    Args:
        query (str): 用户查询
        document (str): 文档文本

    Returns:
        float: 相关性评分（0 到 1）
    """
    # 定义系统提示语，指导模型如何评估相关性
    system_prompt = """
    你是一位评估文档相关性的专家。
    请在 0 到 1 的范围内对给定文档与查询的相关性进行评分。
    0 表示完全不相关，1 表示完全相关。
    仅返回一个介于 0 和 1 之间的浮点数评分，不要过多解释与生成。
    """

    # 构造用户提示语，包含查询和文档内容
    user_prompt = f"查询：{query}\n\n文档：{document}"

    try:
        # 调用 OpenAI API 进行相关性评分
        response = client.chat.completions.create(
            model=llm_model,  # 使用的模型
            messages=[
                {"role": "system", "content": system_prompt},  # 系统消息用于引导助手行为
                {"role": "user", "content": user_prompt}  # 用户消息包含查询和文档
            ],
            temperature=0,  # 设置生成温度为最低以保证一致性
            max_tokens=5  # 只需返回一个简短的分数
        )

        # 提取评分结果
        score_text = response.choices[0].message.content.strip()
        # 使用正则表达式提取响应中的浮点数值
        score_match = re.search(r'(\d+(\.\d+)?)', score_text)
        if score_match:
            return float(score_match.group(1))  # 返回提取到的浮点型评分
        return 0.5  # 如果解析失败，默认返回中间值

    except Exception as e:
        # 捕获异常并打印错误信息，出错时返回默认值
        print(f"评估文档相关性时出错：{e}")
        return 0.5  # 出错时默认返回中等评分
```

## 网络搜索函数


```python
def duck_duck_go_search(query, num_results=3):
    """
    使用 DuckDuckGo 执行网络搜索。

    Args:
        query (str): 搜索查询语句
        num_results (int): 要返回的结果数量

    Returns:
        Tuple[str, List[Dict]]: 合并后的搜索结果文本 和 来源元数据
    """
    # 对查询进行URL编码
    encoded_query = quote_plus(query)

    # DuckDuckGo 的非官方 API 接口地址
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"

    try:
        # 发送网络搜索请求
        response = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        data = response.json()

        # 初始化变量用于存储搜索结果和来源信息
        results_text = ""
        sources = []

        # 添加摘要内容（如果存在）
        if data.get("AbstractText"):
            results_text += f"{data['AbstractText']}\n\n"
            sources.append({
                "title": data.get("AbstractSource", "Wikipedia"),
                "url": data.get("AbstractURL", "")
            })

        # 添加相关主题搜索结果
        for topic in data.get("RelatedTopics", [])[:num_results]:
            if "Text" in topic and "FirstURL" in topic:
                results_text += f"{topic['Text']}\n\n"
                sources.append({
                    "title": topic.get("Text", "").split(" - ")[0],
                    "url": topic.get("FirstURL", "")
                })

        return results_text, sources

    except Exception as e:
        # 如果主搜索失败，打印错误信息
        print(f"执行网络搜索时出错：{e}")

        # 尝试使用备份的搜索API（如SerpAPI）
        try:
            backup_url = f"https://serpapi.com/search.json?q={encoded_query}&engine=duckduckgo"
            response = requests.get(backup_url)
            data = response.json()

            # 初始化变量
            results_text = ""
            sources = []

            # 从备份API提取结果
            for result in data.get("organic_results", [])[:num_results]:
                results_text += f"{result.get('title', '')}: {result.get('snippet', '')}\n\n"
                sources.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", "")
                })

            return results_text, sources
        except Exception as backup_error:
            # 如果备份搜索也失败，打印错误并返回空结果
            print(f"备用搜索也失败了：{backup_error}")
            return "无法获取搜索结果。", []
```


```python
def rewrite_search_query(query):
    """
    将查询重写为更适合网络搜索的形式。

    Args:
        query (str): 原始查询语句

    Returns:
        str: 重写后的查询语句
    """
    # 定义系统提示，指导模型如何重写查询
    system_prompt = """
    你是一位编写高效搜索查询的专家。
    请将给定的查询重写为更适合搜索引擎的形式。
    重点使用关键词和事实，去除不必要的词语，使查询更简洁明确。
    """

    try:
        # 调用 OpenAI API 来重写查询
        response = client.chat.completions.create(
            model=llm_model,  # 使用的模型
            messages=[
                {"role": "system", "content": system_prompt},  # 系统提示用于引导助手行为
                {"role": "user", "content": f"原始查询：{query}\n\n重写后的查询："}  # 用户输入原始查询
            ],
            temperature=0.3,  # 设置生成温度以控制输出随机性
            max_tokens=50  # 限制响应长度
        )

        # 返回重写后的查询结果（去除首尾空白）
        return response.choices[0].message.content.strip()

    except Exception as e:
        # 如果发生错误，打印错误信息并返回原始查询
        print(f"重写搜索查询时出错：{e}")
        return query  # 出错时返回原始查询
```


```python
def perform_web_search(query):
    """
    使用重写后的查询执行网络搜索。

    Args:
        query (str): 用户原始查询语句

    Returns:
        Tuple[str, List[Dict]]: 搜索结果文本 和 来源元数据列表
    """
    # 重写查询以提升搜索效果
    rewritten_query = rewrite_search_query(query)
    print(f"重写后的搜索查询：{rewritten_query}")

    # 使用重写后的查询执行网络搜索
    results_text, sources = duck_duck_go_search(rewritten_query)

    # 返回搜索结果和来源信息
    return results_text, sources
```

## 知识提炼函数



```python
def refine_knowledge(text):
    """
    从文本中提取并精炼关键信息。

    Args:
        text (str): 要精炼的输入文本

    Returns:
        str: 精炼后的关键要点
    """
    # 定义系统提示，指导模型如何提取关键信息
    system_prompt = """
    请从以下文本中提取关键信息，并以清晰简洁的项目符号列表形式呈现。
    重点关注最相关和最重要的事实与细节。
    你的回答应格式化为一个项目符号列表，每一项以 "• " 开头，换行分隔。
    """

    try:
        # 调用 OpenAI API 来精炼文本
        response = client.chat.completions.create(
            model=llm_model,  # 使用的模型
            messages=[
                {"role": "system", "content": system_prompt},  # 系统消息用于引导助手行为
                {"role": "user", "content": f"要提炼的文本内容：\n\n{text}"}  # 用户消息包含待精炼的文本
            ],
            temperature=0.3  # 设置生成温度以控制输出随机性
        )

        # 返回精炼后的关键要点（去除首尾空白）
        return response.choices[0].message.content.strip()

    except Exception as e:
        # 如果发生错误，打印错误信息并返回原始文本
        print(f"精炼知识时出错：{e}")
        return text  # 出错时返回原始文本
```

## CRAG 核心处理


```python
def crag_process(query, vector_store, k=3):
    """
    执行“纠正性检索增强生成”（Corrective RAG）流程。

    Args:
        query (str): 用户查询内容
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        k (int): 初始要检索的文档数量

    Returns:
        Dict: 处理结果，包括响应内容和调试信息
    """
    print(f"\n=== 正在使用 CRAG 处理查询：{query} ===\n")

    # 步骤 1: 创建查询嵌入并检索文档
    print("正在检索初始文档...")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=k)

    # 步骤 2: 评估文档相关性
    print("正在评估文档的相关性...")
    relevance_scores = []
    for doc in retrieved_docs:
        score = evaluate_document_relevance(query, doc["text"])
        relevance_scores.append(score)
        doc["relevance"] = score
        print(f"文档得分为 {score:.2f} 的相关性")

    # 步骤 3: 根据最高相关性得分确定操作策略
    max_score = max(relevance_scores) if relevance_scores else 0
    best_doc_idx = relevance_scores.index(max_score) if relevance_scores else -1

    # 记录来源用于引用
    sources = []
    final_knowledge = ""

    # 步骤 4: 根据情况执行相应的知识获取策略
    if max_score > 0.7:
        # 情况 1: 高相关性 - 直接使用文档内容
        print(f"高相关性 ({max_score:.2f}) - 直接使用文档内容")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        final_knowledge = best_doc
        sources.append({
            "title": "文档",
            "url": ""
        })

    elif max_score < 0.3:
        # 情况 2: 低相关性 - 使用网络搜索
        print(f"低相关性 ({max_score:.2f}) - 进行网络搜索")
        web_results, web_sources = perform_web_search(query)
        final_knowledge = refine_knowledge(web_results)
        sources.extend(web_sources)

    else:
        # 情况 3: 中等相关性 - 结合文档与网络搜索结果
        print(f"中等相关性 ({max_score:.2f}) - 结合文档与网络搜索")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        refined_doc = refine_knowledge(best_doc)

        # 获取网络搜索结果
        web_results, web_sources = perform_web_search(query)
        refined_web = refine_knowledge(web_results)

        # 合并知识
        final_knowledge = f"来自文档的内容:\n{refined_doc}\n\n来自网络搜索的内容:\n{refined_web}"

        # 添加来源
        sources.append({
            "title": "文档",
            "url": ""
        })
        sources.extend(web_sources)

    # 步骤 5: 生成最终响应
    print("正在生成最终响应...")
    response = generate_response(query, final_knowledge, sources)

    # 返回完整的处理结果
    return {
        "query": query,
        "response": response,
        "retrieved_docs": retrieved_docs,
        "relevance_scores": relevance_scores,
        "max_relevance": max_score,
        "final_knowledge": final_knowledge,
        "sources": sources
    }
```

## 生成回答


```python
def generate_response(query, knowledge, sources):
    """
    根据查询内容和提供的知识生成回答。

    Args:
        query (str): 用户的查询内容
        knowledge (str): 用于生成回答的知识内容
        sources (List[Dict]): 来源列表，每个来源包含标题和URL

    Returns:
        str: 生成的回答文本
    """

    # 将来源格式化为可用于提示的内容
    sources_text = ""
    for source in sources:
        title = source.get("title", "未知来源")
        url = source.get("url", "")
        if url:
            sources_text += f"- {title}: {url}\n"
        else:
            sources_text += f"- {title}\n"

    # 定义系统指令（system prompt），指导模型如何生成回答
    system_prompt = """
    你是一个乐于助人的AI助手。请根据提供的知识内容，生成一个全面且有信息量的回答。
    在回答中包含所有相关信息，同时保持语言清晰简洁。
    如果知识内容不能完全回答问题，请指出这一限制。
    最后在回答末尾注明引用来源。
    """

    # 构建用户提示（user prompt），包含用户的查询、知识内容和来源信息
    user_prompt = f"""
    查询内容：{query}

    知识内容：
    {knowledge}

    引用来源：
    {sources_text}

    请根据以上信息，提供一个有帮助的回答，并在最后列出引用来源。
    """

    try:
        # 调用 OpenAI API 生成回答
        response = client.chat.completions.create(
            model=llm_model,  # 使用模型以获得高质量回答
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2  # 控制生成内容的随机性（较低值更稳定）
        )

        # 返回生成的回答内容，并去除首尾空格
        return response.choices[0].message.content.strip()

    except Exception as e:
        # 捕获异常并返回错误信息
        print(f"生成回答时出错: {e}")
        return f"抱歉，在尝试回答您的问题“{query}”时遇到了错误。错误信息为：{str(e)}"
```

## 评估函数


```python
def evaluate_crag_response(query, response, reference_answer=None):
    """
    评估 CRAG 回答的质量。

    Args:
        query (str): 用户查询内容
        response (str): 生成的回答内容
        reference_answer (str, optional): 参考答案（用于对比）

    Returns:
        Dict: 包含评分指标的字典
    """

    # 定义系统指令（system prompt），指导模型如何评估回答质量
    system_prompt = """
    你是评估问答质量的专家。请根据以下标准对提供的回答进行评分：

    1. 相关性 (0-10)：回答是否直接针对查询？
    2. 准确性 (0-10)：信息是否事实正确？
    3. 完整性 (0-10)：回答是否全面覆盖查询的所有方面？
    4. 清晰度 (0-10)：回答是否清晰易懂？
    5. 来源质量 (0-10)：回答是否恰当引用相关来源？

    请以 JSON 格式返回每个维度的评分和简要说明。
    同时包含一个 "overall_score" (0-10) 和简短的 "summary" 总结评估结果。
    """

    # 构建用户提示（user prompt），包含查询和待评估的回答
    user_prompt = f"""
    查询内容：{query}

    待评估的回答：
    {response}
    """

    # 如果提供了参考答案，则将其加入提示词中
    if reference_answer:
        user_prompt += f"""
    参考答案（用于对比）：
    {reference_answer}
    """

    try:
        # 调用模型进行评估
        evaluation_response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},  # 要求返回 JSON 格式
            temperature=0  # 设置为 0 表示完全确定性输出
        )

        # 解析模型返回的评估结果
        evaluation = json.loads(evaluation_response.choices[0].message.content)
        return evaluation

    except Exception as e:
        # 处理评估过程中的异常情况
        print(f"评估回答时出错: {e}")
        return {
            "error": str(e),
            "overall_score": 0,
            "summary": "由于发生错误，评估失败。"
        }
```


```python
def compare_crag_vs_standard_rag(query, vector_store, reference_answer=None):
    """
    比较 CRAG 与标准 RAG 在给定查询上的表现。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        reference_answer (str, optional): 用于比较的参考答案

    Returns:
        Dict: 比较结果，包含查询、CRAG 响应、标准 RAG 响应、评估结果等
    """
    # 运行 CRAG 流程
    print("\n=== 正在运行 CRAG ===")
    crag_result = crag_process(query, vector_store)
    crag_response = crag_result["response"]

    # 运行标准 RAG（直接检索并生成响应）
    print("\n=== 正在运行标准 RAG ===")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=3)
    combined_text = "\n\n".join([doc["text"] for doc in retrieved_docs])
    standard_sources = [{"title": "Document", "url": ""}]
    standard_response = generate_response(query, combined_text, standard_sources)

    # 评估两种方法的结果
    print("\n=== 正在评估 CRAG 响应 ===")
    crag_eval = evaluate_crag_response(query, crag_response, reference_answer)

    print("\n=== 正在评估标准 RAG 响应 ===")
    standard_eval = evaluate_crag_response(query, standard_response, reference_answer)

    # 对比两种方法的表现
    print("\n=== 正在对比两种方法 ===")
    comparison = compare_responses(query, crag_response, standard_response, reference_answer)

    return {
        "query": query,
        "crag_response": crag_response,
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "crag_evaluation": crag_eval,
        "standard_evaluation": standard_eval,
        "comparison": comparison
    }

```


```python
def compare_responses(query, crag_response, standard_response, reference_answer=None):
    """
    比较 CRAG 和标准 RAG 的生成回答。

    Args:
        query (str): 用户查询内容
        crag_response (str): CRAG 方法生成的回答
        standard_response (str): 标准 RAG 方法生成的回答
        reference_answer (str, optional): 参考答案（用于对比）

    Returns:
        str: 对比分析结果
    """

    # 定义系统指令（system prompt），指导模型如何比较两种方法
    system_prompt = """
    你是评估问答系统的专家，请对以下两种方法进行比较分析：

    1. **CRAG**（纠正性检索增强生成）：会先评估文档相关性，并在必要时动态切换至网络搜索的方法。
    2. **标准 RAG**（传统检索增强生成）：基于嵌入向量相似性直接检索文档并生成回答。

    请从以下维度进行比较分析这两种方法的回答：
    - **准确性**：事实内容是否正确？
    - **相关性**：回答是否紧扣查询问题？
    - **完整性**：是否覆盖了问题的所有方面？
    - **清晰度**：语言组织是否清晰易懂？
    - **来源质量**：引用是否合理可靠？

    最后需说明哪种方法在此特定查询中表现更优，并解释原因。
    """

    # 构建用户提示（user prompt），包含查询和两种回答
    user_prompt = f"""
    查询内容：{query}

    CRAG 回答：
    {crag_response}

    标准 RAG 回答：
    {standard_response}
    """

    # 如果提供了参考答案，则将其加入提示词中
    if reference_answer:
        user_prompt += f"""
    参考答案（用于对比）：
    {reference_answer}
    """

    try:
        # 调用模型进行对比分析
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0  # 设置为 0 表示输出确定性结果
        )

        # 返回模型生成的对比分析结果
        return response.choices[0].message.content.strip()

    except Exception as e:
        # 处理对比过程中的异常情况
        print(f"比较回答时出错: {e}")
        return f"比较回答时出错：{str(e)}"
```

## 完整的评估流程


```python
def run_crag_evaluation(pdf_path, test_queries, reference_answers=None):
    """
    运行 CRAG 在多个测试查询上的完整评估流程。

    Args:
        pdf_path (str): PDF 文档的文件路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], optional): 每个查询对应的标准答案（用于对比）

    Returns:
        Dict: 包含所有评估结果的字典
    """

    # 处理文档并创建向量数据库
    vector_store = process_document(pdf_path)

    results = []  # 存储每个查询的评估结果

    # 遍历所有测试查询
    for i, query in enumerate(test_queries):
        print(f"\n\n===== 正在评估第 {i+1}/{len(test_queries)} 个查询 =====")
        print(f"查询内容：{query}")

        # 获取当前查询的参考答案（如果提供）
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]

        # 执行 CRAG 与标准 RAG 的对比评估
        result = compare_crag_vs_standard_rag(query, vector_store, reference)
        results.append(result)  # 保存单次评估结果

        # 显示本次对比结果
        print("\n=== 对比结果 ===")
        print(result["comparison"])

    # 根据所有单次评估结果生成整体分析报告
    overall_analysis = generate_overall_analysis(results)

    # 返回完整评估结果
    return {
        "results": results,              # 单次查询评估结果列表
        "overall_analysis": overall_analysis  # 整体分析报告
    }
```


```python
def generate_overall_analysis(results):
    """
    根据单次查询评估结果生成整体分析报告。

    Args:
        results (List[Dict]): 来自多次查询评估的结果数据

    Returns:
        str: 整体分析报告文本
    """

    # 系统指令（system prompt），指导模型如何生成整体分析
    system_prompt = """
你是信息检索与回答生成系统的评估专家。请基于多个测试查询提供整体分析，对比 CRAG（纠正性 RAG）与标准 RAG 方法。

需重点关注以下内容：
1. **CRAG 的优势场景**：列举并解释 CRAG 表现优于标准 RAG 的情况及原因
2. **标准 RAG 的优势场景**：列举并解释标准 RAG 更优的情况及原因
3. **方法对比总结**：归纳两种方法的核心优缺点
4. **应用建议**：提出针对不同场景的推荐使用方案

要求分析具体、有深度，并结合实际测试数据说明结论。
"""

    # 构建评估结果摘要（供大模型参考）
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"第 {i+1} 个查询：{result['query']}\n"

        if 'crag_evaluation' in result and 'overall_score' in result['crag_evaluation']:
            crag_score = result['crag_evaluation'].get('overall_score', 'N/A')
            evaluations_summary += f"CRAG 综合评分：{crag_score}\n"

        if 'standard_evaluation' in result and 'overall_score' in result['standard_evaluation']:
            std_score = result['standard_evaluation'].get('overall_score', 'N/A')
            evaluations_summary += f"标准 RAG 综合评分：{std_score}\n"

        evaluations_summary += f"对比摘要：{result['comparison'][:200]}...\n\n"

    # 用户指令（user prompt），请求生成分析
    user_prompt = f"""
    基于以下包含 {len(results)} 个查询的 CRAG 与标准 RAG 对比评估结果，请提供这两种方法的整体分析：

    {evaluations_summary}

    请全面分析 CRAG 相对于标准 RAG 的优劣势，重点说明在哪些场景下某种方法更优及其原因。
    """

    try:
        # 调用模型生成整体分析
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0  # 设置为 0 保证输出确定性
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        # 处理分析生成过程中的异常
        print(f"生成整体分析时出错: {e}")
        return f"生成整体分析失败：{str(e)}"
```

## 用测试查询评估 CRAG


```python
# 使用多个与人工智能相关的查询运行全面评估
test_queries = [
    "机器学习与传统编程有何不同？",
]

# 可选参考答案，用于提升评估质量
reference_answers = [
    "机器学习不同于传统编程，它让计算机从数据中学习模式，而不是遵循明确的指令。在传统编程中，开发人员编写具体的规则供计算机执行，而在机器学习中……"
]

# 运行完整的CRAG与标准RAG对比评估
evaluation_results = run_crag_evaluation(pdf_path, test_queries, reference_answers)

# 打印整体分析结果
print("\n=== CRAG 与 标准 RAG 的整体分析 ===")
print(evaluation_results["overall_analysis"])

```

    正在从 data/AI_Information.en.zh-CN.pdf 提取文本...
    共创建了 13 个文本块
    正在为文本块生成嵌入...
    已创建包含 13 个文本块的向量库
    
    
    ===== 正在评估第 1/1 个查询 =====
    查询内容：机器学习与传统编程有何不同？
    
    === 正在运行 CRAG ===
    
    === 正在使用 CRAG 处理查询：机器学习与传统编程有何不同？ ===
    
    正在检索初始文档...
    正在评估文档的相关性...
    文档得分为 0.50 的相关性
    文档得分为 0.10 的相关性
    文档得分为 0.20 的相关性
    中等相关性 (0.50) - 结合文档与网络搜索
    重写后的搜索查询：机器学习 vs 传统编程 区别
    执行网络搜索时出错：HTTPSConnectionPool(host='api.duckduckgo.com', port=443): Max retries exceeded with url: /?q=%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0+vs+%E4%BC%A0%E7%BB%9F%E7%BC%96%E7%A8%8B+%E5%8C%BA%E5%88%AB&format=json (Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x00000266C84692E0>, 'Connection to api.duckduckgo.com timed out. (connect timeout=None)'))
    正在生成最终响应...
    
    === 正在运行标准 RAG ===
    
    === 正在评估 CRAG 响应 ===
    评估回答时出错: Expecting value: line 1 column 1 (char 0)
    
    === 正在评估标准 RAG 响应 ===
    评估回答时出错: Expecting value: line 1 column 1 (char 0)
    
    === 正在对比两种方法 ===
    
    === 对比结果 ===
    ### 比较分析：CRAG vs. 标准 RAG
    
    #### 1. **准确性**
    - **CRAG**：回答内容准确，涵盖了机器学习与传统编程的核心差异，如方法、数据依赖性、适应性等。所有陈述均符合事实，无明显错误。
    - **标准 RAG**：同样准确，但更侧重于核心方法和输入输出的对比，部分细节（如“深度学习通过神经网络层自动提取特征”）略显技术性，可能对非专业读者不够友好。
    
    **结论**：两者均准确，但CRAG的表述更通俗易懂。
    
    #### 2. **相关性**
    - **CRAG**：回答紧扣问题，从多个维度（方法、数据、适应性等）展开对比，完全围绕“机器学习与传统编程的不同”这一主题。
    - **标准 RAG**：相关性也较高，但部分内容（如“透明度”部分）虽然相关，但略微偏离核心问题（差异对比）。
    
    **结论**：CRAG更紧密围绕问题核心。
    
    #### 3. **完整性**
    - **CRAG**：覆盖了问题的所有关键方面，包括方法、数据依赖性、适应性、应用场景、开发流程和错误处理，甚至提到两者的结合使用。
    - **标准 RAG**：缺少对数据依赖性和开发流程的讨论，但补充了“透明度”这一额外维度。
    
    **结论**：CRAG更全面，标准RAG略有遗漏。
    
    #### 4. **清晰度**
    - **CRAG**：语言组织清晰，分点明确，逻辑流畅，易于理解。例如，直接对比“传统编程”和“机器学习”的子条目。
    - **标准 RAG**：表述清晰，但部分术语（如“XAI”）可能增加理解难度，且对比结构不如CRAG直观。
    
    **结论**：CRAG更清晰易懂。
    
    #### 5. **来源质量**
    - **CRAG**：引用来源仅标注“文档”，未说明具体文献或权威性。
    - **标准 RAG**：引用《理解人工智能》第一章至第六章，来源更具体，可能更具权威性。
    
    **结论**：标准RAG的引用更可靠，但CRAG的内容质量未受影响。
    
    ---
    
    ### 综合评估
    **CRAG在此查询中表现更优**，原因如下：
    1. **更全面的对比**：CRAG覆盖了更多关键差异（如数据依赖性、开发流程），而标准RAG遗漏了部分内容。
    2. **更高的相关性**：CRAG完全聚焦于差异对比，而标准RAG引入了“透明度”等次要内容。
    3. **更清晰的表述**：CRAG的语言组织和逻辑结构更易于理解，适合更广泛的读者。
    
    虽然标准RAG的引用来源更具体，但CRAG在准确性、相关性和完整性上的优势更为显著，更适合回答这一查询。
    
    === CRAG 与 标准 RAG 的整体分析 ===
    ### 全面分析：CRAG vs. 标准 RAG
    
    #### 1. **CRAG 的优势场景**
    CRAG（纠正性 RAG）在以下场景中表现优于标准 RAG：
    - **复杂或模糊查询**：当用户查询涉及多义性、隐含逻辑或需要上下文推理时，CRAG 的纠正机制能更好地识别和修正潜在误解。例如，若用户问“机器学习与传统编程有何不同？但不要提数据依赖性”，CRAG 能动态过滤无关内容，而标准 RAG 可能仍保留冗余信息。
    - **动态知识更新需求**：若数据源中存在过时或冲突信息（如新旧研究结论矛盾），CRAG 的置信度评估和外部知识验证能力可优先选择最新或权威答案。例如，回答“当前最优的神经网络架构”时，CRAG 可能通过实时检索排除过时方案。
    - **高精度要求领域**：在医疗、法律等容错率低的领域，CRAG 的主动纠错能力（如验证统计数据的来源）能减少幻觉或错误传播。
    
    **原因**：CRAG 通过置信度分数（如低分触发重新检索）和外部知识校准机制，实现了对生成内容的动态质量控制。
    
    ---
    
    #### 2. **标准 RAG 的优势场景**
    标准 RAG 在以下场景更优：
    - **简单事实型查询**：对于明确、结构化的问题（如“Python 的创始人是谁？”），标准 RAG 直接检索-生成的流程效率更高，无需额外纠正开销。
    - **实时性要求高**：当响应速度比绝对准确性更重要时（如聊天机器人对话），标准 RAG 的轻量级流程更具优势。例如，回答“机器学习定义”时，标准 RAG 可能比 CRAG 快 20-30%。
    - **资源受限环境**：CRAG 的纠正机制需要额外计算（如多轮检索验证），在边缘设备或低算力场景下，标准 RAG 更可行。
    
    **原因**：标准 RAG 的端到端设计减少了中间步骤，牺牲部分纠错能力换取速度和资源效率。
    
    ---
    
    #### 3. **方法对比总结**
    | **维度**       | **CRAG**                          | **标准 RAG**                     |
    |----------------|-----------------------------------|----------------------------------|
    | **准确性**     | 更高（主动纠错）                  | 中等（依赖检索质量）             |
    | **响应速度**   | 较慢（需验证步骤）                | 更快（直接生成）                 |
    | **适用查询**   | 复杂、动态、高精度需求            | 简单、明确、实时性需求           |
    | **资源消耗**   | 高（多轮检索/验证）               | 低（单轮流程）                   |
    | **抗幻觉能力** | 强（外部知识校准）                | 弱（受限于初始检索）             |
    
    ---
    
    #### 4. **应用建议**
    - **推荐 CRAG 的场景**：  
      - 专业领域问答（如学术研究、技术文档分析）  
      - 需要结合实时数据的决策支持（如金融趋势预测）  
      - 存在争议性或多版本答案的问题（如“新冠病毒传播途径的演变”）  
    
    - **推荐标准 RAG 的场景**：  
      - 大众化百科类问答（如“爱因斯坦的生平”）  
      - 实时对话系统（如客服机器人）  
      - 嵌入式设备或低延迟需求应用  
    
    **测试数据佐证**：  
    在开放域问答测试集（如 Natural Questions）中，CRAG 在复杂问题的准确率比标准 RAG 高 8-12%，但响应时间增加 40%；而在简单事实类问题（如 TriviaQA）上，两者准确率差异不足 2%，但标准 RAG 速度快 50%。
    
    --- 
    
    **总结**：选择取决于任务需求——CRAG 是“质量优先”的解决方案，标准 RAG 是“效率优先”的默认选项。混合架构（如对高置信答案直接生成，低置信时触发 CRAG）可能是平衡方案。


## 本地无法调用 duckduckgo时（需要魔法），可以用serpapi调用

- https://serpapi.com/ ，创建一个账号，然后获取API key，每月免费100次请求
