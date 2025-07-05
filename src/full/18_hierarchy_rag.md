# 用于RAG的分级索引

实现一种用于RAG系统的分级索引方法(Hierarchical Indices)。这种技术通过使用两级搜索方法来提高检索效果：首先通过摘要识别相关的文档部分，然后从这些部分中检索具体细节。

-----
传统的RAG方法将所有文本块一视同仁，这可能导致：

- 当文本块过小时，上下文信息丢失
- 当文档集合较大时，检索结果无关
- 在整个语料库中搜索效率低下

-----
分级检索解决了这些问题，具体方式如下：

- 为较大的文档部分创建简洁的摘要
- 首先搜索这些摘要以确定相关部分
- 然后仅从这些部分中检索详细信息
- 在保留具体细节的同时保持上下文信息

-----
实现步骤：
- 从 PDF 中提取页面
- 为每一页创建摘要，将摘要文本和元数据添加到摘要列表中
- 为每一页创建详细块，将页面的文本切分为块
- 为以上两个创建嵌入，并行其存入向量存储中
- 使用查询分层检索相关块：先检索相关的摘要，收集来自相关摘要的页面，然后过滤掉不是相关页面的块，从这些相关页面检索详细块
- 根据检索到的块生成回答


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
import matplotlib.pyplot as plt
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from typing import List, Dict, Tuple, Any
import pickle

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

```


```python
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

```

## 向量存储


```python
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

```

## 创建嵌入


```python
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

```

## 摘要函数


```python
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

```

## 分级文档处理


```python
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
        print(f"正在摘要第 {i+1}/{len(pages)} 页...")
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

```

## 分级检索


```python
def retrieve_hierarchically(query, summary_store, detailed_store, k_summaries=3, k_chunks=5):
    """
    使用分层索引检索信息。

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

```

## 利用上下文生成回答


```python
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

```

## 用分级检索实现完整的RAG流程


```python
def hierarchical_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200, k_summaries=3, k_chunks=5, regenerate=False):
    """
    完整的分层 RAG 管道。

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

```

## 标准 RAG（非分级，用于对比）


```python
def standard_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200, k=15):
    """
    标准 RAG 管道，不包含分层检索。

    Args:
        query (str): 用户查询
        pdf_path (str): PDF 文档的路径
        chunk_size (int): 每个块的大小
        chunk_overlap (int): 块之间的重叠
        k (int): 要检索的块数量

    Returns:
        Dict: 包括响应和检索到的块的结果
    """
    # 从 PDF 文档中提取页面
    pages = extract_text_from_pdf(pdf_path)

    # 直接从所有页面创建块
    chunks = []
    for page in pages:
        # 将页面的文本切分为块
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        # 将当前页面的块扩展到块列表中
        chunks.extend(page_chunks)

    print(f"为标准 RAG 创建了 {len(chunks)} 个块")

    # 创建一个向量存储以保存块
    store = SimpleVectorStore()

    # 为块创建嵌入
    print("正在为块创建嵌入...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    # 将块添加到向量存储中
    for i, chunk in enumerate(chunks):
        store.add_item(
            text=chunk["text"],
            embedding=embeddings[i],
            metadata=chunk["metadata"]
        )

    # 为查询创建嵌入
    query_embedding = create_embeddings(query)

    # 根据查询嵌入检索最相关的块
    retrieved_chunks = store.similarity_search(query_embedding, k=k)
    print(f"通过标准 RAG 检索到 {len(retrieved_chunks)} 个块")

    # 根据检索到的块生成响应
    response = generate_response(query, retrieved_chunks)

    # 返回结果，包括查询、响应和检索到的块
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks
    }

```

## 评估函数


```python
def compare_approaches(query, pdf_path, reference_answer=None):
    """
    比较分层和标准 RAG 方法。

    Args:
        query (str): 用户查询
        pdf_path (str): PDF 文档的路径
        reference_answer (str, 可选): 用于评估的参考答案

    Returns:
        Dict: 比较结果
    """
    print(f"\n=== 对于查询 {query} 比较 RAG 方法 ===")

    # 运行分层 RAG
    print("\n运行分层 RAG...")
    hierarchical_result = hierarchical_rag(query, pdf_path)
    hier_response = hierarchical_result["response"]

    # 运行标准 RAG
    print("\n运行标准 RAG...")
    standard_result = standard_rag(query, pdf_path)
    std_response = standard_result["response"]

    # 比较分层和标准 RAG 的结果
    comparison = compare_responses(query, hier_response, std_response, reference_answer)

    # 返回包含比较结果的字典
    return {
        "query": query,  # 原始查询
        "hierarchical_response": hier_response,  # 分层 RAG 的响应
        "standard_response": std_response,  # 标准 RAG 的响应
        "reference_answer": reference_answer,  # 用于评估的参考答案
        "comparison": comparison,  # 比较分析
        "hierarchical_chunks_count": len(hierarchical_result["retrieved_chunks"]),  # 分层 RAG 检索到的块数量
        "standard_chunks_count": len(standard_result["retrieved_chunks"])  # 标准 RAG 检索到的块数量
    }

```


```python
def compare_responses(query, hierarchical_response, standard_response, reference=None):
    """
    比较分层和标准 RAG 的响应。

    Args:
        query (str): 用户查询
        hierarchical_response (str): 分层 RAG 的响应
        standard_response (str): 标准 RAG 的响应
        reference (str, 可选): 参考答案

    Returns:
        str: 比较分析
    """
    # 定义系统提示，指导模型如何评估响应
    system_prompt = """你是一个信息检索系统的专业评估者。
请比较针对同一查询的两个回答，一个使用分级检索生成，另一个使用标准检索生成。

请从以下方面进行评估：
1. 准确性：哪个回答提供了更多事实准确的信息？
2. 全面性：哪个回答更好地涵盖了查询的所有方面？
3. 连贯性：哪个回答在逻辑流程和组织结构上更清晰合理？
4. 页码引用：是否有哪个回答更有效地利用了页码引用？

请具体分析每种方法的优势与不足。"""


    # 创建包含查询和两种响应的用户提示
    user_prompt = f"""查询: {query}

分级 RAG 的回答:
{hierarchical_response}

标准 RAG 的回答:
{standard_response}"""

    # 如果提供了参考答案，则将其包含在用户提示中
    if reference:
        user_prompt += f"""

参考答案:
{reference}"""

    # 添加最终指示到用户提示中
    user_prompt += """

请对这两个回答进行详细比较，指出哪种方法表现更好，并说明原因。"""

    # 向 OpenAI API 发送请求以生成比较分析
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息以指导助手
            {"role": "user", "content": user_prompt}  # 用户消息包含查询和响应
        ],
        temperature=0  # 设置响应生成的温度
    )

    # 返回生成的比较分析
    return response.choices[0].message.content

```


```python
def run_evaluation(pdf_path, test_queries, reference_answers=None):
    """
    运行带有多个测试查询的完整评估。

    Args:
        pdf_path (str): PDF 文档的路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], 可选): 查询的参考答案列表

    Returns:
        Dict: 评估结果
    """
    results = []  # 初始化一个空列表以存储结果

    # 遍历测试查询中的每个查询
    for i, query in enumerate(test_queries):
        print(f"Query: {query}")  # 打印当前查询

        # 如果可用，获取参考答案
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]  # 获取当前查询的参考答案

        # 比较分层和标准 RAG 方法
        result = compare_approaches(query, pdf_path, reference)
        results.append(result)  # 将结果添加到结果列表中

    # 生成评估结果的整体分析
    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,  # 返回单个结果
        "overall_analysis": overall_analysis  # 返回整体分析
    }

```


```python
def generate_overall_analysis(results):
    """
    生成对评估结果的整体分析。

    Args:
        results (List[Dict]): 来自单个查询评估的结果列表

    Returns:
        str: 整体分析
    """
    # 定义系统提示，指导模型如何评估结果
    system_prompt = """你是一个信息检索系统的专业评估专家。
基于多个测试查询，提供一个整体分析，比较分级RAG与标准RAG的表现。

关注点包括：
1. 分级检索在何时表现更好及其原因
2. 标准检索在何时表现更好及其原因
3. 每种方法的整体优缺点
4. 对于何时使用哪种方法的建议"""

    # 创建评估结果的摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"查询 {i+1}: {result['query']}\n"
        evaluations_summary += f"分级检索使用的文本块数: {result['hierarchical_chunks_count']}, 标准检索使用的文本块数: {result['standard_chunks_count']}\n"
        evaluations_summary += f"比较摘要: {result['comparison'][:200]}...\n\n"

    # 定义用户提示，包含评估摘要内容
    user_prompt = f"""根据以下针对 {len(results)} 个查询的评估结果，比较分级RAG与标准RAG，
请提供这两种方法的整体分析：

{evaluations_summary}

请详细分析分级RAG与标准RAG在检索质量和回答生成方面的相对优缺点，
并提供具体分析。"""

    # 调用 OpenAI API 生成整体分析
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息，用于引导助手的行为
            {"role": "user", "content": user_prompt}       # 用户消息，包含评估摘要
        ],
        temperature=0  # 设置响应生成的随机性（温度参数）
    )

    # 返回生成的整体分析
    return response.choices[0].message.content
```

## 分级RAG与标准RAG方法的评估


```python
# 用于测试分级RAG方法的示例查询
query = "Transformer模型在自然语言处理中的关键应用有哪些？"
result = hierarchical_rag(query, pdf_path)

print("\n=== 回答 ===")
print(result["response"])

# 正式评估使用的测试查询（仅使用一个查询以满足要求）
test_queries = [
    "Transformer是如何处理序列数据的，与RNN相比有何不同？"
]

# 测试查询的参考答案，用于进行比较
reference_answers = [
    "Transformer通过自注意力机制而非循环连接来处理序列数据，这使得Transformer可以并行处理所有token，而不是像RNN那样按顺序处理。这种方法更高效地捕捉长距离依赖关系，并在训练期间实现更好的并行化。与RNN不同，Transformer在处理长序列时不会出现梯度消失的问题。"
]

# 运行评估，比较分级RAG与标准RAG方法
evaluation_results = run_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

# 打印对两种方法的整体分析
print("\n=== 整体分析 ===")
print(evaluation_results["overall_analysis"])
```

    处理文档并创建向量存储...
    正在提取文本 data/AI_Information.en.zh-CN.pdf...
    已提取 15 页的内容
    生成页面摘要...
    正在摘要第 1/15 页...
    正在摘要第 2/15 页...
    正在摘要第 3/15 页...
    正在摘要第 4/15 页...
    正在摘要第 5/15 页...
    正在摘要第 6/15 页...
    正在摘要第 7/15 页...
    正在摘要第 8/15 页...
    正在摘要第 9/15 页...
    正在摘要第 10/15 页...
    正在摘要第 11/15 页...
    正在摘要第 12/15 页...
    正在摘要第 13/15 页...
    正在摘要第 14/15 页...
    正在摘要第 15/15 页...
    已创建 15 个详细块
    正在为摘要创建嵌入...
    正在为详细块创建嵌入...
    已创建包含 15 个摘要和 15 个块的向量存储
    正在为查询执行分层检索: Transformer模型在自然语言处理中的关键应用有哪些？
    检索到 3 个相关摘要
    从相关页面检索到 3 个详细块
    
    === 回答 ===
    根据提供的上下文内容，没有明确提到"Transformer模型"的相关信息（Page 1-3均未提及该术语）。上下文主要介绍了自然语言处理（NLP）作为人工智能的一个分支（Page 2），其应用包括聊天机器人、机器翻译、文本摘要和情感分析等，但并未具体说明这些应用是否由Transformer模型实现。
    
    建议补充Transformer模型相关的上下文内容，或确认是否需要基于现有信息回答。当前可确认的是：
    1. NLP的通用应用领域已在Page 2列出
    2. 深度学习（包含神经网络）是NLP的基础技术之一（Page 2）
    3. 但未涉及Transformer这一特定架构的说明
    Query: Transformer是如何处理序列数据的，与RNN相比有何不同？
    
    === 对于查询 Transformer是如何处理序列数据的，与RNN相比有何不同？ 比较 RAG 方法 ===
    
    运行分层 RAG...
    加载现有的向量存储...
    正在为查询执行分层检索: Transformer是如何处理序列数据的，与RNN相比有何不同？
    检索到 3 个相关摘要
    从相关页面检索到 3 个详细块
    
    运行标准 RAG...
    正在提取文本 data/AI_Information.en.zh-CN.pdf...
    已提取 15 页的内容
    为标准 RAG 创建了 15 个块
    正在为块创建嵌入...
    通过标准 RAG 检索到 15 个块
    
    === 整体分析 ===
    ### 分级RAG与标准RAG的对比分析（基于示例查询）
    
    #### **1. 检索质量对比**
    - **分级RAG优势**：  
      - **精准性**：通过动态调整检索范围（本例仅用3个文本块），优先选择高置信度片段，避免低相关性内容污染上下文。  
      - **容错性**：当高层级（如粗粒度检索）未命中关键信息时，明确承认知识盲区（如直接说明"缺乏Transformer的具体信息"），而非强行生成。  
      - **效率**：减少无关文本处理开销，尤其适合**明确边界的问题**（如需要对比特定技术细节时）。  
    
    - **标准RAG劣势**：  
      - **噪声引入**：强制检索固定数量文本块（本例15个），可能混入低质量内容（如Page 12的间接推断），导致生成答案时被迫"脑补"。  
      - **过度泛化**：试图用宽泛上下文填补细节缺失（如将RNN的序列处理缺陷间接套用到Transformer），增加事实性错误风险。  
    
    #### **2. 回答生成对比**
    - **分级RAG特点**：  
      - **保守但可靠**：生成策略与检索结果严格对齐，缺少直接证据时选择"知之为知之"（如示例中的诚实声明），适合**高事实性要求场景**（学术、医疗等）。  
      - **解释性**：可通过分级逻辑向用户说明检索过程（如"未找到足够细粒度数据"），增强可信度。  
    
    - **标准RAG特点**：  
      - **覆盖性优先**：倾向于利用所有检索内容生成看似完整的答案，但可能包含未经验证的关联（如将RNN的缺陷与Transformer优势强行对比）。  
      - **流畅性陷阱**：因上下文更庞杂，生成的答案往往更长、更"流畅"，但可能掩盖逻辑漏洞（如示例中的间接推断问题）。  
    
    #### **3. 关键场景适用性**
    - **优先选择分级RAG**：  
      - 问题需要**精确技术细节**（如算法对比、参数说明）  
      - 数据源存在**质量不均**或**领域专业性高**（如法律、医学文献）  
      - 用户容忍"部分回答"但要求**零幻觉**  
    
    - **优先选择标准RAG**：  
      - 问题偏向**概述性**或**多角度讨论**（如"深度学习的优缺点"）  
      - 数据源质量均匀且**冗余度高**（如维基百科类文本）  
      - 用户更看重答案**连贯性**而非绝对精确  
    
    #### **4. 改进方向建议**
    - **分级RAG**：可增加"分级置信度"指标（如标注"本回答基于Top 3可信片段，覆盖度70%"），平衡保守性与实用性。  
    - **标准RAG**：需引入**断言验证机制**（如对"Transformer并行处理"等关键说法检查直接引文），减少间接推断。  
    
    **总结**：本例中分级RAG的准确性优势凸显了其在技术性查询中的价值，而标准RAG的"尽力回答"策略在开放性问题上可能更友好。选择取决于任务的核心需求——**事实优先选分级，覆盖优先选标准**。

