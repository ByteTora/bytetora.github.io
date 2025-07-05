# 增强 RAG 系统的上下文压缩技术

上下文情境压缩技术（Contextual Compression），以提高 RAG 系统的效率。过滤并压缩检索到的文本块，只保留最相关的内容，从而减少噪声并提高响应质量。

在为 RAG 检索文档时，经常得到包含相关和不相关信息的块。上下文压缩可以帮助我们：

- 删除无关的句子和段落
- 仅关注与查询相关的信息
- 在上下文窗口中最大化有用信号

本文提供了三种方法：
1. 过滤（selective）：分析文档块并仅提取与用户查询直接相关的句子或段落，移除所有无关内容。
2. 摘要（summary）：创建文档块的简洁摘要，且仅聚焦与用户查询相关的信息。
3. 抽取（extraction）：从文档块中精确提取与用户查询相关的完整句子。

-----
实现步骤：
- 处理文档以创建向量存储：从PDF 中提取文本，分割文本块并创建向量存储
- 创建查询嵌入并检索文档，检索最相似的前k个块
- 对检索到的块应用压缩：
    - 过滤（selective）：分析文档块并仅提取与用户查询直接相关的句子或段落，移除所有无关内容。
    - 摘要（summary）：创建文档块的简洁摘要，且仅聚焦与用户查询相关的信息
    - 抽取（extraction）：从文档块中精确提取与用户查询相关的完整句子
- 过滤掉任何空的压缩块
- 基于压缩块形成上下文内容，然后生成回答



```python
import fitz
import os
import re
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

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


```python
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
```


```python
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
```


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

```


```python
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
```


```python
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

```

## 实现上下文压缩
这是我们的方法的核心——我们将使用大语言模型 (LLM) 来过滤和压缩检索到的内容。


```python
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

```

## 实现批量压缩
为了提高效率，在尽可能的情况下一次性压缩多个文本块。


```python
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
        print(f"正在压缩文本块 {i+1}/{len(chunks)}...")  # 打印压缩进度
        # 压缩文本块并获取压缩后的文本块和压缩比率
        compressed_chunk, compression_ratio = compress_chunk(chunk, query, compression_type)
        results.append((compressed_chunk, compression_ratio))  # 将结果添加到结果列表中

        total_original_length += len(chunk)  # 将原始文本块的长度加到总原始长度中
        total_compressed_length += len(compressed_chunk)  # 将压缩后文本块的长度加到总压缩长度中

    # 计算总体压缩比率
    overall_ratio = (total_original_length - total_compressed_length) / total_original_length * 100
    print(f"总体压缩比率: {overall_ratio:.2f}%")  # 打印总体压缩比率

    return results  # 返回包含压缩文本块和压缩比率的列表

```

## 生成回答


```python
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

```

## 上下文压缩的完整 RAG 管道


```python
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
        "context_length_reduction": f"{sum(compression_ratios)/len(compression_ratios):.2f}%",
        "response": response
    }

    print("\n=== RESPONSE ===")
    print(response)

    return result

```

## 标准RAG与压缩增强型RAG的对比分析
构建对比函数实现标准RAG与压缩增强型RAG的性能比较：


```python
def standard_rag(pdf_path, query, k=10):
    """
    标准RAG，不包含压缩。

    Args:
        pdf_path (str): PDF文档的路径
        query (str): 用户查询
        k (int): 检索的块数量

    Returns:
        dict: 包括查询、块和响应的结果
    """
    print("\n=== STANDARD RAG ===")
    print(f"Query: {query}")

    # 处理文档以提取文本、分块并创建嵌入
    vector_store = process_document(pdf_path)

    # 为查询创建嵌入
    query_embedding = create_embeddings(query)

    # 根据查询嵌入检索最相似的前k个块
    print(f"Retrieving top {k} chunks...")
    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]

    # 从检索到的块生成上下文
    context = "\n\n---\n\n".join(retrieved_chunks)

    # 基于检索到的块生成响应
    print("Generating response...")
    response = generate_response(query, context)

    # 准备结果字典
    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }

    print("\n=== RESPONSE ===")
    print(response)

    return result

```




```python
def evaluate_responses(query, responses, reference_answer):
    """
    评估多个响应与参考答案的对比。

    Args:
        query (str): 用户查询
        responses (Dict[str, str]): 按方法分类的响应字典
        reference_answer (str): 参考答案

    Returns:
        str: 评估文本
    """
    # 定义系统提示，指导AI的行为进行评估
    system_prompt = """您是RAG系统回答内容的客观评估专家。请对比分析同一查询的不同回答，判断哪项回答最精准、最全面且与查询最相关。"""

    # 通过组合查询和参考答案创建用户提示
    user_prompt = f"""
    查询: {query}

    参考答案: {reference_answer}

    """

    # 将每个响应添加到提示中
    for method, response in responses.items():
        user_prompt += f"\n{method.capitalize()} 回答内容:\n{response}\n"

    # 在用户提示中添加评估标准
    user_prompt += """
    请基于以下维度评估各项回答：
    1. 事实准确性（对照参考答案）
    2. 回答完整度（是否全面解答问题）
    3. 内容精简度（是否避免无关信息）
    4. 综合质量

    具体要求：
    - 对所有回答进行排序（从最优到最差）
    - 提供详细的评估依据
    - 指出各项回答的优缺点
    - 最终推荐最优解决方案
    """

    # 使用AI API生成评估响应
    evaluation_response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 返回响应中的评估文本
    return evaluation_response.choices[0].message.content

```


```python
def evaluate_compression(pdf_path, query, reference_answer=None, compression_types=["selective", "summary", "extraction"]):
    """
    比较不同的压缩技术与标准RAG。

    Args:
        pdf_path (str): PDF文档的路径
        query (str): 用户查询
        reference_answer (str): 可选的参考答案
        compression_types (List[str]): 要评估的压缩类型列表

    Returns:
        dict: 评估结果
    """
    print("\n=== 正在评估上下文压缩 ===")
    print(f"查询: {query}")

    # 运行标准RAG（无压缩）
    standard_result = standard_rag(pdf_path, query)

    # 存储不同压缩技术结果的字典
    compression_results = {}

    # 使用每种压缩技术运行RAG
    for comp_type in compression_types:
        print(f"\n正在测试 {comp_type} 压缩...")
        compression_results[comp_type] = rag_with_compression(pdf_path, query, compression_type=comp_type)

    # 收集响应以进行评估
    responses = {
        "standard": standard_result["response"]
    }
    for comp_type in compression_types:
        responses[comp_type] = compression_results[comp_type]["response"]

    # 如果提供了参考答案，则评估响应
    if reference_answer:
        evaluation = evaluate_responses(query, responses, reference_answer)
        print("\n=== 评估结果 ===")
        print(evaluation)
    else:
        evaluation = "未提供参考答案进行评估。"

    # 计算每种压缩类型的指标
    metrics = {}
    for comp_type in compression_types:
        metrics[comp_type] = {
            "avg_compression_ratio": f"{sum(compression_results[comp_type]['compression_ratios'])/len(compression_results[comp_type]['compression_ratios']):.2f}%",
            "total_context_length": len("\n\n".join(compression_results[comp_type]['compressed_chunks'])),
            "original_context_length": len("\n\n".join(standard_result['chunks']))
        }

    # 返回评估结果、响应和指标
    return {
        "query": query,
        "responses": responses,
        "evaluation": evaluation,
        "metrics": metrics,
        "standard_result": standard_result,
        "compression_results": compression_results
    }

```

## 运行我们的完整系统（自定义查询）


```python
# 从文档中提取相关信息的查询
query = "人工智能在决策应用中的伦理有哪些问题？"

# 可选的参考答案，用于评估
reference_answer = """
人工智能在决策中的应用引发多个伦理问题：
- AI模型偏见可能导致不公正或歧视性结果，在招聘、信贷发放及执法等关键领域尤为突出
- AI驱动中的黑箱决策机制使得个体难以质疑不公正结果，透明度和可解释性不足
- AI系统处理海量个人数据时存在隐私泄露风险，通常缺乏明确授权
- 自动化导致的岗位流失引发社会经济层面的担忧
- AI决策权可能集中于少数科技巨头，导致问责机制失效
- 确保AI系统的公平性、问责机制和系统透明度是实现伦理部署的必要条件
"""

# 使用不同的压缩技术进行评估
# 压缩类型：
# - "selective": 保留关键细节，省略不太相关的内容
# - "summary": 提供信息的简洁版本
# - "extraction": 从文档中逐字提取相关句子
results = evaluate_compression(
    pdf_path=pdf_path,  # PDF文件路径
    query=query,  # 查询内容
    reference_answer=reference_answer,  # 参考答案
    compression_types=["selective", "summary", "extraction"]  # 压缩类型列表
)

```

    
    === 正在评估上下文压缩 ===
    查询: 人工智能在决策应用中的伦理有哪些问题？
    
    === STANDARD RAG ===
    Query: 人工智能在决策应用中的伦理有哪些问题？
    从PDF中提取文本...
    分割文本...
    创建了 13 个文本块
    为文本块创建嵌入向量...
    向向量存储中添加了 13 个文本块
    Retrieving top 10 chunks...
    Generating response...
    
    === RESPONSE ===
    根据上下文内容，人工智能在决策应用中的伦理问题主要包括以下几个方面：
    
    ### 1. **偏见与公平性**
    - **问题**：AI系统可能继承并放大训练数据中存在的偏见，导致歧视性或不公平的决策结果。例如，在招聘、信贷审批或司法预测中，历史数据中的偏见可能导致对特定群体的不公平对待。
    - **解决方向**：需通过谨慎的数据收集、算法设计优化（如去偏技术）以及持续的监测评估来减少偏见。
    
    ### 2. **透明度和可解释性**
    - **问题**：许多AI系统（尤其是深度学习模型）是“黑匣子”，其决策过程难以理解。缺乏透明度会削弱用户信任，并妨碍对错误决策的追责。
    - **解决方向**：发展可解释人工智能（XAI）技术，通过可视化、简化模型或生成决策理由来增强可解释性。
    
    ### 3. **隐私和数据安全**
    - **问题**：AI决策依赖大量数据，可能涉及敏感信息（如医疗记录、财务数据）。不当的数据处理或泄露会侵犯隐私权。
    - **解决方向**：采用隐私保护技术（如差分隐私、联邦学习），并遵守数据保护法规（如GDPR）。
    
    ### 4. **问责与责任归属**
    - **问题**：当AI决策导致负面后果（如自动驾驶事故或医疗误诊），责任主体不明确（开发者、部署者或用户）。
    - **解决方向**：需建立明确的法律框架，界定各方的角色与责任，并确保AI系统具备可追溯性。
    
    ### 5. **自主性与人类控制**
    - **问题**：高度自主的AI系统可能脱离人类监管，产生不可预测的后果（如军事武器系统失控）。
    - **解决方向**：制定开发指导方针（如“人类监督”原则），确保关键决策中保留人类否决权。
    
    ### 6. **社会影响与工作岗位流失**
    - **问题**：AI自动化决策可能取代人类岗位（如客服、制造业），引发经济不平等和社会动荡。
    - **解决方向**：推动劳动力再培训计划，促进人机协作模式，并探索AI创造的新就业机会。
    
    ### 7. **伦理框架与全球协作**
    - **问题**：不同地区对AI伦理的认知和标准存在差异，可能导致监管冲突或“伦理洼地”。
    - **解决方向**：推动国际合作（如制定全球性AI伦理准则），确保跨文化、跨领域的伦理共识。
    
    ### 总结
    解决这些伦理问题需要多管齐下，包括技术创新（如XAI）、政策监管（如数据保护法）、行业自律（如伦理审查委员会）以及公众教育。上下文中强调的“负责任AI”原则（公平、透明、隐私、问责）是核心指导方针，而全球协作与持续研究是长期保障。
    
    正在测试 selective 压缩...
    
    === RAG WITH CONTEXTUAL COMPRESSION ===
    Query: 人工智能在决策应用中的伦理有哪些问题？
    Compression type: selective
    从PDF中提取文本...
    分割文本...
    创建了 13 个文本块
    为文本块创建嵌入向量...
    向向量存储中添加了 13 个文本块
    Retrieving top 10 chunks...
    正在压缩 10 个文本块...
    正在压缩文本块 1/10...
    正在压缩文本块 2/10...
    正在压缩文本块 3/10...
    正在压缩文本块 4/10...
    正在压缩文本块 5/10...
    正在压缩文本块 6/10...
    正在压缩文本块 7/10...
    正在压缩文本块 8/10...
    正在压缩文本块 9/10...
    正在压缩文本块 10/10...
    总体压缩比率: 78.22%
    Generating response based on compressed chunks...
    
    === RESPONSE ===
    根据上下文内容，人工智能在决策应用中的伦理问题主要包括以下几个方面：
    
    ### 1. **偏见与公平性**
       - **问题**：人工智能系统可能继承并放大训练数据中的偏见，导致不公平或歧视性决策结果。
       - **表现**：例如在招聘、信贷审批或司法决策中，算法可能因历史数据中的偏见而歧视特定群体。
       - **解决方向**：需通过谨慎的数据收集、算法设计优化以及持续的监测和评估来减少偏见。
    
    ### 2. **透明度和可解释性**
       - **问题**：许多AI决策系统（如深度学习模型）是“黑匣子”，难以理解其决策逻辑。
       - **风险**：缺乏透明度会削弱用户信任，并妨碍对错误决策的追责。
       - **解决方案**：开发可解释人工智能（XAI）技术，使决策过程更透明易懂，确保用户能评估其公平性和准确性。
    
    ### 3. **隐私和数据安全**
       - **问题**：AI决策依赖大量数据，可能侵犯个人隐私或引发数据滥用风险。
       - **挑战**：例如医疗或金融领域的敏感信息处理需高度谨慎。
       - **应对措施**：需实施隐私保护技术、遵守数据保护法规（如GDPR），并确保数据处理的透明性。
    
    ### 4. **自主性与控制**
       - **问题**：高度自主的AI系统可能脱离人类控制，导致意外后果或责任归属不清。
       - **案例**：自动驾驶汽车的道德决策（如“电车难题”）或军事武器的自主化应用。
       - **伦理框架**：需制定国际法规和道德准则，明确开发者和部署者的责任边界。
    
    ### 5. **问责与责任**
       - **问题**：当AI决策造成危害时，责任主体（开发者、部署者或用户）难以界定。
       - **需求**：需建立明确的问责机制，确保受害者获得救济，并推动伦理合规。
    
    ### 6. **社会影响与权利保护**
       - **问题**：AI决策可能加剧社会不平等（如自动化导致失业）或侵犯人权。
       - **原则**：需以“以人为本的AI”为核心，确保系统符合人类价值观，优先保护隐私、非歧视和福祉。
    
    ### 7. **武器化的伦理风险**
       - **特殊领域**：AI在自主武器中的应用涉及生命权剥夺的伦理困境。
       - **国际共识**：需通过全球合作禁止或严格限制此类用途，防止滥用。
    
    ### 总结性伦理原则
    根据上下文，伦理AI决策应遵循以下核心原则：
    - **公平性**：避免偏见和歧视。
    - **透明性**：确保决策可解释。
    - **隐私保护**：安全处理数据。
    - **问责制**：明确责任划分。
    - **人权尊重**：优先保护人类权益。
    - **社会效益**：以促进福祉为目标。
    
    这些问题的解决需要跨学科合作、国际标准制定以及持续的伦理审查机制。
    
    正在测试 summary 压缩...
    
    === RAG WITH CONTEXTUAL COMPRESSION ===
    Query: 人工智能在决策应用中的伦理有哪些问题？
    Compression type: summary
    从PDF中提取文本...
    分割文本...
    创建了 13 个文本块
    为文本块创建嵌入向量...
    向向量存储中添加了 13 个文本块
    Retrieving top 10 chunks...
    正在压缩 10 个文本块...
    正在压缩文本块 1/10...
    正在压缩文本块 2/10...
    正在压缩文本块 3/10...
    正在压缩文本块 4/10...
    正在压缩文本块 5/10...
    正在压缩文本块 6/10...
    正在压缩文本块 7/10...
    正在压缩文本块 8/10...
    正在压缩文本块 9/10...
    正在压缩文本块 10/10...
    总体压缩比率: 87.67%
    Generating response based on compressed chunks...
    
    === RESPONSE ===
    基于提供的上下文内容，人工智能在决策应用中的伦理问题可系统归纳为以下核心方面：
    
    ---
    
    ### **1. 偏见与公平性**
    - **数据偏见放大**：AI可能继承并放大训练数据中的历史或社会偏见（如种族、性别歧视），导致不公平决策结果（如招聘、信贷审批）。
    - **算法歧视**：设计缺陷或数据选择不当可能使特定群体受到系统性歧视，需通过公平性算法和偏见检测工具缓解。
    
    ---
    
    ### **2. 透明性与可解释性（XAI）**
    - **"黑匣子"问题**：深度学习模型决策过程难以追溯，需发展可解释人工智能（XAI）技术以提高透明度。
    - **信任建立**：用户需理解AI决策逻辑以评估其合理性，尤其在医疗、司法等高风险领域。
    
    ---
    
    ### **3. 隐私与数据安全**
    - **数据依赖风险**：AI依赖大量个人数据，可能违反隐私法规（如GDPR），需匿名化处理和加密技术保护。
    - **滥用潜在**：数据泄露或监控滥用可能侵犯人权，需严格的数据治理框架。
    
    ---
    
    ### **4. 自主性与责任归属**
    - **失控风险**：高度自主的AI系统（如自动驾驶、无人机）可能引发不可预测行为，需明确人类监督机制。
    - **问责空白**：当AI决策造成损害时，责任划分困难（开发者、部署者或用户），需法律明确责任链条。
    
    ---
    
    ### **5. 武器化与安全威胁**
    - **自主武器伦理**：致命性自主武器系统（LAWS）可能绕过人类判断，引发战争伦理争议，需国际公约限制。
    - **恶意使用**：AI可能被用于网络攻击或深度伪造，需全球协作制定安全标准。
    
    ---
    
    ### **6. 就业与社会影响**
    - **劳动力替代**：自动化导致就业流失，需政策干预（如再培训计划）和"人机协作"伦理框架。
    - **工人权利保护**：AI监控员工绩效可能侵犯隐私，需平衡效率与权利。
    
    ---
    
    ### **7. 道德框架与全球治理**
    - **伦理原则缺失**：需制定公平、透明、问责的伦理准则（如OECD AI原则），并嵌入系统设计阶段。
    - **国际合作需求**：跨国标准不一致可能加剧风险，需联合国等机构推动协同治理（如数据主权、算法审计）。
    
    ---
    
    ### **8. 用户权利与人性化设计**
    - **知情与选择权**：用户应有权知晓AI参与决策并选择退出（如个性化推荐）。
    - **以人为本**：AI开发需考量社会心理影响，优先保障人类尊严和福祉（如避免情感操纵）。
    
    ---
    
    ### **总结**
    这些问题相互交织，需多学科协作（技术、法律、哲学）和动态监管解决。核心矛盾在于平衡AI的创新潜力与对人类社会价值观的保护，最终目标是实现**负责任的人工智能（Responsible AI）**。
    
    正在测试 extraction 压缩...
    
    === RAG WITH CONTEXTUAL COMPRESSION ===
    Query: 人工智能在决策应用中的伦理有哪些问题？
    Compression type: extraction
    从PDF中提取文本...
    分割文本...
    创建了 13 个文本块
    为文本块创建嵌入向量...
    向向量存储中添加了 13 个文本块
    Retrieving top 10 chunks...
    正在压缩 10 个文本块...
    正在压缩文本块 1/10...
    正在压缩文本块 2/10...
    正在压缩文本块 3/10...
    正在压缩文本块 4/10...
    正在压缩文本块 5/10...
    正在压缩文本块 6/10...
    正在压缩文本块 7/10...
    正在压缩文本块 8/10...
    正在压缩文本块 9/10...
    正在压缩文本块 10/10...
    总体压缩比率: 83.68%
    Generating response based on compressed chunks...
    
    === RESPONSE ===
    根据上下文，人工智能在决策应用中的伦理问题主要包括以下几个方面：
    
    1. **偏见与公平性**  
       - 人工智能系统可能继承并放大训练数据中的偏见，导致不公平或歧视性结果。  
       - 解决需要谨慎的数据收集、算法设计及持续的监测和评估。
    
    2. **透明度与可解释性**  
       - 许多AI系统（如深度学习模型）是“黑匣子”，决策过程难以理解。  
       - 可解释人工智能（XAI）技术旨在提升透明度，帮助用户评估决策的公平性和准确性。
    
    3. **隐私与数据安全**  
       - AI依赖大量数据，可能引发隐私泄露风险。  
       - 需确保敏感信息保护及负责任的数据处理。
    
    4. **问责制与责任归属**  
       - 随着AI自主性增强，需明确开发者、部署者和用户的责任。  
       - 建立问责机制以应对潜在危害和道德问题。
    
    5. **自主武器系统的伦理争议**  
       - 军事应用引发重大安全担忧，需国际法规约束其风险。
    
    6. **伦理框架与治理缺失**  
       - 当前缺乏统一的道德准则和监管标准，需制定指导方针以确保开发符合人权、非歧视等原则。
    
    7. **社会与心理影响**  
       - 需考量AI对就业、人类福祉的影响，优先以人为中心的设计。
    
    综上，解决这些问题需结合伦理评估、利益相关者协商及跨领域合作，以平衡创新与社会效益。
    
    === 评估结果 ===
    ### 评估结果排序（从最优到最差）：
    1. **Summary 回答内容**
    2. **Standard 回答内容**
    3. **Selective 回答内容**
    4. **Extraction 回答内容**
    
    ---
    
    ### 详细评估依据
    
    #### **1. Summary 回答内容**  
    **优点：**  
    - **事实准确性**：完全覆盖参考答案的所有核心问题（偏见、透明度、隐私、问责等），并补充了参考答案未提及的“武器化与安全威胁”“就业与社会影响”等细分领域。  
    - **回答完整度**：最全面，将伦理问题归纳为8个系统化维度，每个维度包含问题描述、风险案例和解决方向，逻辑清晰。  
    - **内容精简度**：虽内容详细，但无冗余信息，通过分级标题和分点排版提升可读性。  
    - **综合质量**：提出“负责任的人工智能”总结框架，体现跨学科协作和动态监管的前瞻性。  
    
    **缺点**：部分子项（如“武器化”）的解决方向可更具体。
    
    ---
    
    #### **2. Standard 回答内容**  
    **优点：**  
    - **事实准确性**：涵盖参考答案全部要点，额外补充“伦理框架与全球协作”这一重要维度。  
    - **回答完整度**：7个问题分类合理，每个问题均提供“解决方向”，实用性较强。  
    - **内容精简度**：结构清晰，但部分子项（如“社会影响”）的论述略简略。  
    
    **缺点**：相比Summary回答，缺少“武器化”“数据主权”等前沿议题的深入分析。
    
    ---
    
    #### **3. Selective 回答内容**  
    **优点：**  
    - **事实准确性**：覆盖参考答案核心内容，但未提及“岗位流失”等社会经济影响。  
    - **回答完整度**：7个问题分类与Standard类似，但“武器化的伦理风险”独立成项，体现一定创新性。  
    - **内容精简度**：部分子项（如“隐私和数据安全”）重复提及GDPR，略显冗余。  
    
    **缺点**：总结部分仅罗列原则，未提出具体实施路径，完整性稍逊。
    
    ---
    
    #### **4. Extraction 回答内容**  
    **优点：**  
    - **事实准确性**：包含参考答案主要问题，但“自主武器”仅一句话带过，缺乏深度。  
    - **回答完整度**：7点分类较笼统，如“社会与心理影响”未展开就业等具体案例。  
    - **内容精简度**：过于简略，部分要点（如“伦理框架”）仅提及“缺失”，未提供解决建议。  
    
    **缺点**：整体流于表面，缺乏案例和解决方案的实质性内容。
    
    ---
    
    ### 最优解决方案推荐  
    **推荐采用「Summary 回答内容」**，因其具备以下优势：  
    1. **全面性与前瞻性**：覆盖传统伦理问题（如偏见、隐私）和新兴挑战（如武器化、全球治理），符合技术发展趋势。  
    2. **结构化表达**：通过分级标题和分点论述，兼顾专业性与可读性，适合不同受众。  
    3. **解决方案导向**：每个问题均提供缓解措施（如XAI技术、国际公约），可直接指导实践。  
    
    **改进建议**：可补充“伦理审查流程”的具体案例（如欧盟AI法案），进一步增强实操性。


## 可视化压缩结果


```python
def visualize_compression_results(evaluation_results):
    """
    可视化不同压缩技术的结果。

    Args:
        evaluation_results (Dict): 来自 evaluate_compression 函数的结果
    """
    # 从评估结果中提取查询和标准块
    query = evaluation_results["query"]
    standard_chunks = evaluation_results["standard_result"]["chunks"]

    # 打印查询内容
    print(f"Query: {query}")  # 查询内容
    print("\n" + "="*80 + "\n")  # 分隔线

    # 获取一个示例块以进行可视化（使用第一个块）
    original_chunk = standard_chunks[0]

    # 遍历每种压缩类型并显示比较结果
    for comp_type in evaluation_results["compression_results"].keys():
        compressed_chunks = evaluation_results["compression_results"][comp_type]["compressed_chunks"]
        compression_ratios = evaluation_results["compression_results"][comp_type]["compression_ratios"]

        # 获取对应的压缩块及其压缩比率
        compressed_chunk = compressed_chunks[0]
        compression_ratio = compression_ratios[0]

        print(f"\n=== {comp_type.upper()} COMPRESSION EXAMPLE ===\n")  # 压缩类型的标题

        # 显示原始块（如果过长则截断）
        print("ORIGINAL CHUNK:")  # 标题：原始块
        print("-" * 40)  # 分隔线
        if len(original_chunk) > 800:  # 如果原始块长度超过 800 字符，则截断
            print(original_chunk[:800] + "... [truncated]")  # 截断后的文本
        else:
            print(original_chunk)  # 完整的原始块
        print("-" * 40)  # 分隔线
        print(f"Length: {len(original_chunk)} characters\n")  # 原始块的长度

        # 显示压缩块
        print("COMPRESSED CHUNK:")  # 标题：压缩块
        print("-" * 40)  # 分隔线
        print(compressed_chunk)  # 压缩后的文本
        print("-" * 40)  # 分隔线
        print(f"Length: {len(compressed_chunk)} characters")  # 压缩块的长度
        print(f"Compression ratio: {compression_ratio:.2f}%\n")  # 压缩比率

        # 显示该压缩类型的总体统计数据
        avg_ratio = sum(compression_ratios) / len(compression_ratios)  # 平均压缩比率
        print(f"Average compression across all chunks: {avg_ratio:.2f}%")  # 所有块的平均压缩比率
        print(f"Total context length reduction: {evaluation_results['metrics'][comp_type]['avg_compression_ratio']}")  # 总上下文长度减少
        print("=" * 80)  # 分隔线

    # 显示压缩技术的汇总表
    print("\n=== COMPRESSION SUMMARY ===\n")  # 汇总标题
    print(f"{'Technique':<15} {'Avg Ratio':<15} {'Context Length':<15} {'Original Length':<15}")  # 表头
    print("-" * 60)  # 分隔线

    # 打印每种压缩技术的指标
    for comp_type, metrics in evaluation_results["metrics"].items():
        print(f"{comp_type:<15} {metrics['avg_compression_ratio']:<15} {metrics['total_context_length']:<15} {metrics['original_context_length']:<15}")  # 每种技术的详细数据

```


```python
# Visualize the compression results
visualize_compression_results(results)
```

    Query: 人工智能在决策应用中的伦理有哪些问题？
    
    ================================================================================
    
    
    === SELECTIVE COMPRESSION EXAMPLE ===
    
    ORIGINAL CHUNK:
    ----------------------------------------
    问题也随之⽽来。为⼈⼯智能的开发
    和部署建⽴清晰的指导⽅针和道德框架⾄关重要。
    ⼈⼯智能武器化
    ⼈⼯智能在⾃主武器系统中的潜在应⽤引发了重⼤的伦理和安全担忧。需要开展国际讨论并制定相
    关法规，以应对⼈⼯智能武器的相关⻛险。
    第五章：⼈⼯智能的未来
    ⼈⼯智能的未来很可能以持续进步和在各个领域的⼴泛应⽤为特征。关键趋势和发展领域包括：
    可解释⼈⼯智能（XAI）
    可解释⼈⼯智能 (XAI) 旨在使⼈⼯智能系统更加透明易懂。XAI 技术正在开发中，旨在深⼊了解⼈
    ⼯智能模型的决策⽅式，从⽽增强信任度和责任感。
    边缘⼈⼯智能
    边缘⼈⼯智能是指在设备上本地处理数据，⽽不是依赖云服务器。这种⽅法可以减少延迟，增强隐
    私保护，并在连接受限的环境中⽀持⼈⼯智能应⽤。
    量⼦计算和⼈⼯智能
    量⼦计算有望显著加速⼈⼯智能算法，从⽽推动药物研发、材料科学和优化等领域的突破。量⼦计
    算与⼈⼯智能的交叉研究前景⼴阔。
    ⼈机协作
    ⼈⼯智能的未来很可能涉及⼈类与⼈⼯智能系统之间更紧密的协作。这包括开发能够增强⼈类能
    ⼒、⽀持决策和提⾼⽣产⼒的⼈⼯智能⼯具。
    ⼈⼯智能造福社会
    ⼈⼯智能正⽇益被⽤于应对社会和环境挑战，例如⽓候变化、贫困和医疗保健差距。“⼈⼯智能造
    福社会”倡议旨在利⽤⼈⼯智能产⽣积极影响。
    监管与治理
    随着⼈⼯智能⽇益普及，监管和治理的需求将⽇益增⻓，以确保负责任的开发和部署。这包括制定
    道德准则、解决偏⻅和公平问题，以及保护隐私和安全。国际标准合作⾄关重要。
    通过了解⼈⼯智能的核⼼概念、应⽤、伦理影响和未来发展⽅向，我们可以更好地应对这项变⾰性
    技术带来的机遇和挑战。持续的研究、负责任的开发和周到的治理，对于充分发挥⼈⼯智能的潜⼒
    并降低其⻛险⾄关重要。
    第六章：⼈⼯智能和机器⼈技术
    ⼈⼯智能与机器⼈技术的融合
    ⼈⼯智能与机器⼈技术的融合，将机器⼈的物理能⼒与⼈⼯智能的认知能⼒完美结合。这种... [truncated]
    ----------------------------------------
    Length: 1000 characters
    
    COMPRESSED CHUNK:
    ----------------------------------------
    为⼈⼯智能的开发
    和部署建⽴清晰的指导⽅针和道德框架⾄关重要。
    ⼈⼯智能武器化
    ⼈⼯智能在⾃主武器系统中的潜在应⽤引发了重⼤的伦理和安全担忧。需要开展国际讨论并制定相
    关法规，以应对⼈⼯智能武器的相关⻛险。
    可解释⼈⼯智能（XAI）
    可解释⼈⼯智能 (XAI) 旨在使⼈⼯智能系统更加透明易懂。XAI 技术正在开发中，旨在深⼊了解⼈
    ⼯智能模型的决策⽅式，从⽽增强信任度和责任感。
    监管与治理
    随着⼈⼯智能⽇益普及，监管和治理的需求将⽇益增⻓，以确保负责任的开发和部署。这包括制定
    道德准则、解决偏⻅和公平问题，以及保护隐私和安全。国际标准合作⾄关重要。
    ----------------------------------------
    Length: 277 characters
    Compression ratio: 72.30%
    
    Average compression across all chunks: 77.73%
    Total context length reduction: 77.73%
    ================================================================================
    
    === SUMMARY COMPRESSION EXAMPLE ===
    
    ORIGINAL CHUNK:
    ----------------------------------------
    问题也随之⽽来。为⼈⼯智能的开发
    和部署建⽴清晰的指导⽅针和道德框架⾄关重要。
    ⼈⼯智能武器化
    ⼈⼯智能在⾃主武器系统中的潜在应⽤引发了重⼤的伦理和安全担忧。需要开展国际讨论并制定相
    关法规，以应对⼈⼯智能武器的相关⻛险。
    第五章：⼈⼯智能的未来
    ⼈⼯智能的未来很可能以持续进步和在各个领域的⼴泛应⽤为特征。关键趋势和发展领域包括：
    可解释⼈⼯智能（XAI）
    可解释⼈⼯智能 (XAI) 旨在使⼈⼯智能系统更加透明易懂。XAI 技术正在开发中，旨在深⼊了解⼈
    ⼯智能模型的决策⽅式，从⽽增强信任度和责任感。
    边缘⼈⼯智能
    边缘⼈⼯智能是指在设备上本地处理数据，⽽不是依赖云服务器。这种⽅法可以减少延迟，增强隐
    私保护，并在连接受限的环境中⽀持⼈⼯智能应⽤。
    量⼦计算和⼈⼯智能
    量⼦计算有望显著加速⼈⼯智能算法，从⽽推动药物研发、材料科学和优化等领域的突破。量⼦计
    算与⼈⼯智能的交叉研究前景⼴阔。
    ⼈机协作
    ⼈⼯智能的未来很可能涉及⼈类与⼈⼯智能系统之间更紧密的协作。这包括开发能够增强⼈类能
    ⼒、⽀持决策和提⾼⽣产⼒的⼈⼯智能⼯具。
    ⼈⼯智能造福社会
    ⼈⼯智能正⽇益被⽤于应对社会和环境挑战，例如⽓候变化、贫困和医疗保健差距。“⼈⼯智能造
    福社会”倡议旨在利⽤⼈⼯智能产⽣积极影响。
    监管与治理
    随着⼈⼯智能⽇益普及，监管和治理的需求将⽇益增⻓，以确保负责任的开发和部署。这包括制定
    道德准则、解决偏⻅和公平问题，以及保护隐私和安全。国际标准合作⾄关重要。
    通过了解⼈⼯智能的核⼼概念、应⽤、伦理影响和未来发展⽅向，我们可以更好地应对这项变⾰性
    技术带来的机遇和挑战。持续的研究、负责任的开发和周到的治理，对于充分发挥⼈⼯智能的潜⼒
    并降低其⻛险⾄关重要。
    第六章：⼈⼯智能和机器⼈技术
    ⼈⼯智能与机器⼈技术的融合
    ⼈⼯智能与机器⼈技术的融合，将机器⼈的物理能⼒与⼈⼯智能的认知能⼒完美结合。这种... [truncated]
    ----------------------------------------
    Length: 1000 characters
    
    COMPRESSED CHUNK:
    ----------------------------------------
    人工智能在决策应用中的伦理问题包括：
    1. 需要建立道德框架和指导方针
    2. 自主武器系统引发安全担忧
    3. 可解释人工智能(XAI)技术旨在提高决策透明度
    4. 监管需求增加，需制定道德准则、解决偏见和公平问题
    5. 国际标准合作对负责任开发至关重要
    ----------------------------------------
    Length: 126 characters
    Compression ratio: 87.40%
    
    Average compression across all chunks: 87.48%
    Total context length reduction: 87.48%
    ================================================================================
    
    === EXTRACTION COMPRESSION EXAMPLE ===
    
    ORIGINAL CHUNK:
    ----------------------------------------
    问题也随之⽽来。为⼈⼯智能的开发
    和部署建⽴清晰的指导⽅针和道德框架⾄关重要。
    ⼈⼯智能武器化
    ⼈⼯智能在⾃主武器系统中的潜在应⽤引发了重⼤的伦理和安全担忧。需要开展国际讨论并制定相
    关法规，以应对⼈⼯智能武器的相关⻛险。
    第五章：⼈⼯智能的未来
    ⼈⼯智能的未来很可能以持续进步和在各个领域的⼴泛应⽤为特征。关键趋势和发展领域包括：
    可解释⼈⼯智能（XAI）
    可解释⼈⼯智能 (XAI) 旨在使⼈⼯智能系统更加透明易懂。XAI 技术正在开发中，旨在深⼊了解⼈
    ⼯智能模型的决策⽅式，从⽽增强信任度和责任感。
    边缘⼈⼯智能
    边缘⼈⼯智能是指在设备上本地处理数据，⽽不是依赖云服务器。这种⽅法可以减少延迟，增强隐
    私保护，并在连接受限的环境中⽀持⼈⼯智能应⽤。
    量⼦计算和⼈⼯智能
    量⼦计算有望显著加速⼈⼯智能算法，从⽽推动药物研发、材料科学和优化等领域的突破。量⼦计
    算与⼈⼯智能的交叉研究前景⼴阔。
    ⼈机协作
    ⼈⼯智能的未来很可能涉及⼈类与⼈⼯智能系统之间更紧密的协作。这包括开发能够增强⼈类能
    ⼒、⽀持决策和提⾼⽣产⼒的⼈⼯智能⼯具。
    ⼈⼯智能造福社会
    ⼈⼯智能正⽇益被⽤于应对社会和环境挑战，例如⽓候变化、贫困和医疗保健差距。“⼈⼯智能造
    福社会”倡议旨在利⽤⼈⼯智能产⽣积极影响。
    监管与治理
    随着⼈⼯智能⽇益普及，监管和治理的需求将⽇益增⻓，以确保负责任的开发和部署。这包括制定
    道德准则、解决偏⻅和公平问题，以及保护隐私和安全。国际标准合作⾄关重要。
    通过了解⼈⼯智能的核⼼概念、应⽤、伦理影响和未来发展⽅向，我们可以更好地应对这项变⾰性
    技术带来的机遇和挑战。持续的研究、负责任的开发和周到的治理，对于充分发挥⼈⼯智能的潜⼒
    并降低其⻛险⾄关重要。
    第六章：⼈⼯智能和机器⼈技术
    ⼈⼯智能与机器⼈技术的融合
    ⼈⼯智能与机器⼈技术的融合，将机器⼈的物理能⼒与⼈⼯智能的认知能⼒完美结合。这种... [truncated]
    ----------------------------------------
    Length: 1000 characters
    
    COMPRESSED CHUNK:
    ----------------------------------------
    为⼈⼯智能的开发
    和部署建⽴清晰的指导⽅针和道德框架⾄关重要。
    ⼈⼯智能在⾃主武器系统中的潜在应⽤引发了重⼤的伦理和安全担忧。
    需要开展国际讨论并制定相
    关法规，以应对⼈⼯智能武器的相关⻛险。
    随着⼈⼯智能⽇益普及，监管和治理的需求将⽇益增⻓，以确保负责任的开发和部署。
    这包括制定
    道德准则、解决偏⻅和公平问题，以及保护隐私和安全。
    ----------------------------------------
    Length: 167 characters
    Compression ratio: 83.30%
    
    Average compression across all chunks: 82.91%
    Total context length reduction: 82.91%
    ================================================================================
    
    === COMPRESSION SUMMARY ===
    
    Technique       Avg Ratio       Context Length  Original Length
    ------------------------------------------------------------
    selective       77.73%          2081            9489           
    summary         87.48%          1186            9489           
    extraction      82.91%          1564            9489           

