# RAG中的反馈循环机制(Feedback Loop)
![](../images/Feedback_Loop.webp)
实现一个具备反馈循环机制的RAG系统，通过持续学习实现性能迭代优化。系统将收集并整合用户反馈数据，使每次交互都能提升回答的相关性与质量。

-----
传统RAG系统采用静态检索模式，仅依赖嵌入相似性获取信息。而本系统通过反馈循环构建动态优化框架，实现：
- 记忆有效/无效的交互模式
- 动态调整文档相关性权重
- 将优质问答对整合至知识库
- 通过用户互动持续增强智能水平

-----
实现步骤：
- 加载历史反馈数据集
- 文档预处理与分块处理
- 可选基于历史反馈微调向量索引
- 基于反馈修正的相关性评分执行检索与生成
- 收集新用户反馈数据用于后续优化
- 持久化存储反馈数据支撑系统持续学习能力



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

# 10.1 反馈系统函数

核心反馈系统的组件函数


```python
def get_user_feedback(query, response, relevance, quality, comments=""):
    """
    将用户反馈格式化为字典。

    Args:
        query (str): 用户的查询
        response (str): 系统的回答
        relevance (int): 相关性评分 (1-5)
        quality (int): 质量评分 (1-5)
        comments (str): 可选的反馈评论

    Returns:
        Dict: 格式化的反馈
    """
    return {
        "query": query,
        "response": response,
        "relevance": int(relevance),
        "quality": int(quality),
        "comments": comments,
        "timestamp": datetime.now().isoformat()  # 当前时间戳
    }

```


```python
def store_feedback(feedback, feedback_file="feedback_data.json"):
    """
    将反馈存储在JSON文件中。

    Args:
        feedback (Dict): 反馈数据
        feedback_file (str): 反馈文件的路径
    """
    with open(feedback_file, "a", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=4)
        f.write("\n")

```


```python
def load_feedback_data(feedback_file="feedback_data.json"):
    """
    从文件中加载反馈数据。

    Args:
        feedback_file (str): 反馈文件的路径

    Returns:
        List[Dict]: 反馈条目的列表
    """
    feedback_data = []
    try:
        with open(feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("未找到反馈数据文件。将以空反馈开始。")
        # print("No feedback data file found. Starting with empty feedback.")

    return feedback_data

```

# 10.2 具有反馈认知的文档处理


```python
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    为带有反馈循环的RAG（检索增强生成）处理文档。
    该函数处理完整的文档处理管道：
    1. 从PDF中提取文本
    2. 带有重叠的文本分块
    3. 为每个文本块创建向量嵌入
    4. 在向量数据库中存储带有元数据的块

    Args:
        pdf_path (str): 要处理的PDF文件路径。
        chunk_size (int): 每个文本块的字符数。
        chunk_overlap (int): 相邻块之间的重叠字符数。

    Returns:
        Tuple[List[str], SimpleVectorStore]: 包含以下内容的元组：
            - 文档块列表
            - 填充了嵌入和元数据的向量存储
    """
    # 第一步：从PDF文档中提取原始文本内容
    print("从PDF中提取文本...")
    extracted_text = extract_text_from_pdf(pdf_path)

    # 第二步：将文本分成可管理的、带有重叠的块，以便更好地保存上下文
    print("对文本进行分块...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"创建了 {len(chunks)} 个文本块")

    # 第三步：为每个文本块生成向量嵌入
    print("为文本块创建嵌入...")
    chunk_embeddings = create_embeddings(chunks)

    # 第四步：初始化向量数据库以存储块及其嵌入
    store = SimpleVectorStore()

    # 第五步：将每个块及其嵌入添加到向量存储中
    # 包含用于基于反馈改进的元数据
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={
                "index": i,                # 在原始文档中的位置
                "source": pdf_path,        # 源文档路径
                "relevance_score": 1.0,    # 初始相关性分数（将通过反馈更新）
                "feedback_count": 0        # 接收到此块反馈的计数器
            }
        )

    print(f"已将 {len(chunks)} 个块添加到向量存储中")
    return chunks, store

```

# 10.3 基于反馈的相关性调整


```python
def assess_feedback_relevance(query, doc_text, feedback):
    """
    调用大语言模型（LLM）判定历史反馈条目与当前查询及文档的关联性。

    该函数通过向LLM提交以下内容实现智能判定：
    1. 当前查询语句
    2. 历史查询及对应反馈数据
    3. 关联文档内容
    最终确定哪些历史反馈应影响当前检索优化。

    Args:
        query (str): 当前需要信息检索的用户查询
        doc_text (str): 正在评估的文档文本内容
        feedback (Dict): 包含 'query' 和 'response' 键的过去反馈数据

    Returns:
        bool: 如果反馈被认为与当前查询/文档相关，则返回True，否则返回False
    """
    # 定义系统提示，指示LLM仅进行二元相关性判断
    system_prompt = """您是专门用于判断历史反馈与当前查询及文档相关性的AI系统。
    请仅回答 'yes' 或 'no'。您的任务是严格判断相关性，无需提供任何解释。"""

    # 构造用户提示，包含当前查询、过去的反馈数据以及截断[truncated]的文档内容
    user_prompt = f"""
    当前查询: {query}
    收到反馈的历史查询: {feedback['query']}
    文档内容: {doc_text[:500]}... [截断]
    收到反馈的历史响应: {feedback['response'][:500]}... [truncated]

    该历史反馈是否与当前查询及文档相关？(yes/no)
    """

    # 调用LLM API，设置温度为0以获得确定性输出
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 使用温度=0以确保一致性和确定性响应
    )

    # 提取并规范化响应以确定相关性
    answer = response.choices[0].message.content.strip().lower()
    return 'yes' in answer  # 如果答案中包含 'yes'，则返回True

```


```python
def adjust_relevance_scores(query, results, feedback_data):
    """
    基于历史反馈数据动态调整文档关联分数以优化检索质量。

    本函数通过分析历史用户反馈实现以下优化流程：
    1. 识别与当前查询上下文相关的历史反馈
    2. 根据关联度分数（相关性评分）计算分数修正因子
    3. 基于修正结果重排序检索文档

    Args:
        query (str): 当前用户查询
        results (List[Dict]): 检索到的文档及其原始相似度分数
        feedback_data (List[Dict]): 包含用户评分的历史反馈

    Returns:
        List[Dict]: 调整后的相关性分数结果，按新分数排序
    """
    # 如果没有反馈数据，则返回原始结果不变
    if not feedback_data:
        return results

    print("基于反馈历史调整相关性分数...")

    # 处理每个检索到的文档
    for i, result in enumerate(results):
        document_text = result["text"]
        relevant_feedback = []

        # 查找与此特定文档和查询组合相关的反馈
        # 通过调用LLM评估每个历史反馈项的相关性
        for feedback in feedback_data:
            is_relevant = assess_feedback_relevance(query, document_text, feedback)
            if is_relevant:
                relevant_feedback.append(feedback)

        # 如果存在相关反馈，则应用分数调整
        if relevant_feedback:
            # 计算所有适用反馈条目的平均相关性评分
            # 反馈相关性为1-5分（1=不相关，5=高度相关）
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)

            # 将平均相关性转换为范围在0.5-1.5的分数调整因子
            # - 低于3/5的分数将降低原始相似度（调整因子 < 1.0）
            # - 高于3/5的分数将增加原始相似度（调整因子 > 1.0）
            modifier = 0.5 + (avg_relevance / 5.0)

            # 将调整因子应用于原始相似度分数
            original_score = result["similarity"]
            adjusted_score = original_score * modifier

            # 更新结果字典中的新分数和反馈元数据
            result["original_similarity"] = original_score  # 保留原始分数
            result["similarity"] = adjusted_score         # 更新主分数
            result["relevance_score"] = adjusted_score   # 更新相关性分数
            result["feedback_applied"] = True            # 标记反馈已应用
            result["feedback_count"] = len(relevant_feedback)  # 使用的反馈条目数量

            # 记录调整细节
            print(f"  文档 {i+1}: 基于 {len(relevant_feedback)} 条反馈，分数从 {original_score:.4f} 调整为 {adjusted_score:.4f}")

    # 按调整后的分数重新排序结果，确保更高匹配质量的结果优先显示
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results

```

# 10.4 通过反馈微调我们的索引



```python
def fine_tune_index(current_store, chunks, feedback_data):
    """
    通过高质量反馈数据增强向量存储，实现检索质量的持续优化。

    本函数通过以下机制实现持续学习流程：
    1. 筛选优质反馈数据（高评分问答对）
    2. 将成功交互案例转化为检索条目
    3. 为新增条目配置强化关联权重并注入向量库

    Args:
        current_store (SimpleVectorStore): 当前包含原始文档块的向量存储
        chunks (List[str]): 原始文档文本块
        feedback_data (List[Dict]): 用户的历史反馈数据，包含相关性和质量评分

    Returns:
        SimpleVectorStore: 增强后的向量存储，包含原始块和基于反馈生成的内容
    """
    print("使用高质量反馈微调索引...")

    # 筛选出高质量反馈（相关性和质量评分均达到4或5）
    # 这确保我们仅从最成功的交互中学习
    good_feedback = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]

    if not good_feedback:
        print("未找到可用于微调的高质量反馈。")
        return current_store  # 如果没有高质量反馈，则返回原始存储不变

    # 初始化一个新的存储，它将包含原始内容和增强内容
    new_store = SimpleVectorStore()

    # 首先将所有原始文档块及其现有元数据转移到新存储中
    for i in range(len(current_store.texts)):
        new_store.add_item(
            text=current_store.texts[i],  # 原始文本
            embedding=current_store.vectors[i],  # 对应的嵌入向量
            metadata=current_store.metadata[i].copy()  # 使用副本防止引用问题
        )

    # 根据高质量反馈创建并添加增强内容
    for feedback in good_feedback:
        # 将问题和高质量答案组合成新的文档格式
        # 这样可以创建直接解决用户查询的可检索内容
        enhanced_text = f"Question: {feedback['query']}\nAnswer: {feedback['response']}"

        # 为这个新的合成文档生成嵌入向量
        embedding = create_embeddings(enhanced_text)

        # 将其添加到向量存储中，并附带特殊元数据以标识其来源和重要性
        new_store.add_item(
            text=enhanced_text,
            embedding=embedding,
            metadata={
                "type": "feedback_enhanced",  # 标记为来自反馈生成
                "query": feedback["query"],   # 保存原始查询以供参考
                "relevance_score": 1.2,       # 提高初始相关性以优先处理这些项
                "feedback_count": 1,          # 跟踪反馈整合情况
                "original_feedback": feedback  # 保存完整的反馈记录
            }
        )

        print(f"已添加来自反馈的增强内容: {feedback['query'][:50]}...")

    # 记录关于增强的汇总统计信息
    print(f"微调后的索引现在有 {len(new_store.texts)} 个项目 (原始: {len(chunks)})")
    return new_store

```

# 10.5 使用循环反馈的完整 RAG 管道


```python
def generate_response(query, context):
    """
    根据查询和上下文生成响应。

    Args:
        query (str): 用户查询
        context (str): 从检索文档中提取的上下文文本

    Returns:
        str: 生成的响应
    """
    # 定义系统提示以指导AI的行为
    system_prompt = "您是一个乐于助人的AI助手。请仅根据提供的上下文来回答用户的问题。如果在上下文中找不到答案，请直接说'没有足够的信息'。"

    # 通过结合上下文和查询创建用户提示
    user_prompt = f"""
        上下文:
        {context}

        问题: {query}

        请基于上述上下文内容提供一个全面详尽的答案。
    """

    # 调用OpenAI API，根据系统提示和用户提示生成响应
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 使用temperature=0以获得一致且确定性的响应
    )

    # 返回生成的响应内容
    return response.choices[0].message.content

```


```python
def rag_with_feedback_loop(query, vector_store, feedback_data, k=5):
    """
    完整的RAG管道，包含反馈循环。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        feedback_data (List[Dict]): 反馈历史
        k (int): 检索的文档数量

    Returns:
        Dict: 包括查询、检索到的文档和响应的结果
    """
    print(f"\n=== 使用反馈增强型RAG处理查询 ===")
    print(f"查询: {query}")

    # 第1步：创建查询嵌入
    query_embedding = create_embeddings(query)

    # 第2步：基于查询嵌入执行初始检索
    results = vector_store.similarity_search(query_embedding, k=k)

    # 第3步：根据反馈调整检索到的文档的相关性分数
    adjusted_results = adjust_relevance_scores(query, results, feedback_data)

    # 第4步：从调整后的结果中提取文本以构建上下文
    retrieved_texts = [result["text"] for result in adjusted_results]

    # 第5步：通过连接检索到的文本构建用于生成响应的上下文
    context = "\n\n---\n\n".join(retrieved_texts)

    # 第6步：使用上下文和查询生成响应
    print("正在生成响应...")
    response = generate_response(query, context)

    # 第7步：编译最终结果
    result = {
        "query": query,
        "retrieved_documents": adjusted_results,
        "response": response
    }

    print("\n=== 响应 ===")
    print(response)

    return result

```

# 10.6 完整的工作流程：从初始设置到反馈收集



```python
def full_rag_workflow(pdf_path, query, feedback_data=None, feedback_file="feedback_data.json", fine_tune=False):
    """
    协调执行完整的RAG工作流，集成反馈机制实现持续优化提升。

    本函数系统化执行检索增强生成（RAG）全流程：
    1. 加载历史反馈数据集
    2. 文档预处理与分块处理
    3. 可选基于历史反馈微调向量索引
    4. 基于反馈修正的相关性评分执行检索与生成
    5. 收集新用户反馈数据用于后续优化
    6. 持久化存储反馈数据支撑系统持续学习能力

    Args:
        pdf_path (str): 要处理的PDF文档路径
        query (str): 用户的自然语言查询
        feedback_data (List[Dict], optional): 预加载的反馈数据，如果为None则从文件加载
        feedback_file (str): 存储反馈历史的JSON文件路径
        fine_tune (bool): 是否通过成功的过往问答对来增强索引

    Returns:
        Dict: 包含响应和检索元数据的结果
    """
    # 第1步：如果未明确提供，则加载历史反馈数据以进行相关性调整
    if feedback_data is None:
        feedback_data = load_feedback_data(feedback_file)
        print(f"从 {feedback_file} 加载了 {len(feedback_data)} 条反馈记录")

    # 第2步：通过提取、分块和嵌入管道处理文档
    chunks, vector_store = process_document(pdf_path)

    # 第3步：通过结合高质量的过往交互微调向量索引
    # 这将从成功的问答对中创建增强的可检索内容
    if fine_tune and feedback_data:
        vector_store = fine_tune_index(vector_store, chunks, feedback_data)

    # 第4步：执行核心RAG并使用反馈感知检索
    # 注意：这依赖于rag_with_feedback_loop函数，应在其他地方定义
    result = rag_with_feedback_loop(query, vector_store, feedback_data)

    # 第5步：收集用户反馈以改进未来的表现
    print("\n=== 您是否愿意对这个响应提供反馈？ ===")
    print("评分相关性（1-5，5表示最相关）：")
    relevance = input()

    print("评分质量（1-5，5表示最高质量）：")
    quality = input()

    print("有任何评论吗？（可选，按Enter跳过）")
    comments = input()

    # 第6步：将反馈格式化为结构化数据
    feedback = get_user_feedback(
        query=query,
        response=result["response"],
        relevance=int(relevance),
        quality=int(quality),
        comments=comments
    )

    # 第7步：持久化反馈以实现系统的持续学习
    store_feedback(feedback, feedback_file)
    print("反馈已记录。感谢您的参与！")

    return result

```

# 10.7 评估循环反馈

## 10.7.1 评估过程中的辅助函数


```python
def calculate_similarity(text1, text2):
    """
    使用嵌入向量计算两个文本之间的语义相似度。

    Args:
        text1 (str): 第一个文本
        text2 (str): 第二个文本

    Returns:
        float: 介于 0 和 1 之间的相似度分数
    """
    # 为两个文本生成嵌入向量
    embedding1 = create_embeddings(text1)
    embedding2 = create_embeddings(text2)

    # 将嵌入向量转换为 numpy 数组
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # 计算两个向量之间的余弦相似度
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return similarity

```


```python
def compare_results(queries, round1_results, round2_results, reference_answers=None):
    """
    比较两轮 RAG 的结果。

    Args:
        queries (List[str]): 测试查询列表
        round1_results (List[Dict]): 第一轮结果
        round2_results (List[Dict]): 第二轮结果
        reference_answers (List[str], 可选): 参考答案

    Returns:
        str: 比较分析
    """
    print("\n=== 正在比较结果 ===")

    # 系统提示，用于指导 AI 的评估行为
    system_prompt = """您是RAG系统评估专家，负责比较两个版本的响应质量：
        1. 标准RAG系统：未使用反馈机制
        2. 反馈增强型RAG：采用反馈循环优化检索

        请从以下维度分析各版本表现：
        - 与查询的相关性
        - 信息准确性
        - 回答完整性
        - 表述清晰度与简洁性
    """

    comparisons = []

    # 遍历每个查询及其对应的两轮结果
    for i, (query, r1, r2) in enumerate(zip(queries, round1_results, round2_results)):
        # 创建用于比较响应的提示
        comparison_prompt = f"""
        查询: {query}

        标准RAG系统响应:
        {r1["response"]}

        反馈增强型RAG响应:
        {r2["response"]}
        """

        # 如果有参考答案，则包含参考答案
        if reference_answers and i < len(reference_answers):
            comparison_prompt += f"""
            参考答案:
            {reference_answers[i]}
            """

        comparison_prompt += """
        请对比分析两个版本的响应质量，重点说明：
        - 哪个版本表现更优及其原因
        - 反馈循环机制对响应质量的提升效果（或未体现改进的原因）
        """

        # 调用 OpenAI API 生成比较分析
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": comparison_prompt}
            ],
            temperature=0
        )

        # 将比较分析添加到结果中
        comparisons.append({
            "query": query,
            "analysis": response.choices[0].message.content
        })

        # 打印每个查询的分析片段
        print(f"\n查询 {i+1}: {query}")
        print(f"分析: {response.choices[0].message.content[:200]}...")

    return comparisons

```

## 10.7.2 评估函数


```python
def evaluate_feedback_loop(pdf_path, test_queries, reference_answers=None):
    """
    通过对比反馈集成前后的系统表现，评估反馈循环机制对RAG质量的提升效果。

    本函数执行对照实验以量化反馈机制的影响：
    1. 初始阶段（第一轮）：无反馈状态下执行全部测试查询
    2. 生成阶段：基于参考答案创建模拟反馈数据（如有提供）
    3. 增强阶段（第二轮）：采用反馈优化后的检索机制复测相同查询集
    4. 效果分析：跨阶段结果对比实现反馈价值的量化分析

    Args:
        pdf_path (str): 用作知识库的PDF文档路径。
        test_queries (List[str]): 用于评估系统性能的测试查询列表。
        reference_answers (List[str], optional): 用于评估和生成合成反馈的参考/标准答案。

    Returns:
        Dict: 评估结果包含以下内容：
            - round1_results: 无反馈的结果
            - round2_results: 含反馈的结果
            - comparison: 两轮之间的定量比较指标
    """
    print("=== 评估反馈循环的影响 ===")

    # 创建仅用于本次评估会话的临时反馈文件
    temp_feedback_file = "temp_evaluation_feedback.json"

    # 初始化反馈收集（开始时为空）
    feedback_data = []

    # ----------------------- 第一评估轮次 -----------------------
    # 在没有任何反馈影响的情况下运行所有查询，以建立基准性能
    print("\n=== 第一轮：无反馈 ===")
    round1_results = []

    for i, query in enumerate(test_queries):
        print(f"\n查询 {i+1}: {query}")

        # 处理文档以创建初始向量存储
        chunks, vector_store = process_document(pdf_path)

        # 在没有反馈影响的情况下执行RAG（空反馈列表）
        result = rag_with_feedback_loop(query, vector_store, [])
        round1_results.append(result)

        # 如果有参考答案，则生成合成反馈
        # 这模拟了用户反馈以训练系统
        if reference_answers and i < len(reference_answers):
            # 根据与参考答案的相似性计算合成反馈评分
            similarity_to_ref = calculate_similarity(result["response"], reference_answers[i])
            # 将相似性（0-1）转换为评分尺度（1-5）
            relevance = max(1, min(5, int(similarity_to_ref * 5)))
            quality = max(1, min(5, int(similarity_to_ref * 5)))

            # 创建结构化的反馈条目
            feedback = get_user_feedback(
                query=query,
                response=result["response"],
                relevance=relevance,
                quality=quality,
                comments=f"基于参考相似性的合成反馈: {similarity_to_ref:.2f}"
            )

            # 添加到内存中的集合并持久化到临时文件
            feedback_data.append(feedback)
            store_feedback(feedback, temp_feedback_file)

    # ----------------------- 第二评估轮次 -----------------------
    # 在包含反馈的情况下运行相同的查询，以衡量改进
    print("\n=== 第二轮：含反馈 ===")
    round2_results = []

    # 处理文档并增强反馈衍生内容
    chunks, vector_store = process_document(pdf_path)
    vector_store = fine_tune_index(vector_store, chunks, feedback_data)

    for i, query in enumerate(test_queries):
        print(f"\n查询 {i+1}: {query}")

        # 在反馈影响下执行RAG
        result = rag_with_feedback_loop(query, vector_store, feedback_data)
        round2_results.append(result)

    # ----------------------- 结果分析 -----------------------
    # 比较两轮之间的性能指标
    comparison = compare_results(test_queries, round1_results, round2_results, reference_answers)

    # 清理临时评估工件
    if os.path.exists(temp_feedback_file):
        os.remove(temp_feedback_file)

    return {
        "round1_results": round1_results,
        "round2_results": round2_results,
        "comparison": comparison
    }

```

# 10.8 循环反馈评估（自定义验证查询）



```python
# 定义测试查询
test_queries = [
    "什么是神经网络以及它如何工作？",

    #################################################################################
    ### 注释掉的查询以减少测试目的的查询数量 ###

    # "描述强化学习的过程和应用。",
    # "当今技术中自然语言处理的主要应用是什么？",
    # "解释机器学习模型中过拟合的影响以及如何缓解。"
]

# 定义参考答案以进行评估
reference_answers = [
    "神经网络是一系列试图通过模仿人脑运作方式来识别数据集中潜在关系的算法。它由多层节点组成，每个节点代表一个神经元。神经网络通过根据输出误差与预期结果之间的差异调整节点之间的连接权重来运行。",

    ############################################################################################
    #### 注释掉的参考答案以减少测试目的的查询数量 ###

#     "强化学习是一种机器学习类型，代理通过在环境中执行动作以最大化累积奖励来学习做出决策。它涉及探索、利用和从行动后果中学习。应用包括机器人技术、游戏和自动驾驶汽车。",
#     "当今技术中自然语言处理的主要应用包括机器翻译、情感分析、聊天机器人、信息检索、文本摘要和语音识别。NLP 使机器能够理解和生成人类语言，促进人机交互。",
#     "当模型过于学习训练数据时，机器学习模型中会发生过拟合，捕获噪声和异常值。这导致对新数据的泛化能力较差，因为模型在训练数据上表现良好但在未见数据上表现不佳。缓解技术包括交叉验证、正则化、剪枝和使用更多训练数据。"
]

# 运行评估
evaluation_results = evaluate_feedback_loop(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)
```

    === 评估反馈循环的影响 ===
    
    === 第一轮：无反馈 ===
    
    查询 1: 什么是神经网络以及它如何工作？
    从PDF中提取文本...
    对文本进行分块...
    创建了 13 个文本块
    为文本块创建嵌入...
    已将 13 个块添加到向量存储中
    
    === 使用反馈增强型RAG处理查询 ===
    查询: 什么是神经网络以及它如何工作？
    正在生成响应...
    
    === 响应 ===
    根据上下文，神经网络（尤其是人工神经网络）是深度学习的核心组成部分，其设计灵感来源于人脑的结构和功能。以下是关于神经网络及其工作原理的详细说明：
    
    ### 1. **神经网络的定义**
    神经网络是一种由多层互联的“神经元”（或节点）组成的计算模型，用于分析和学习数据中的复杂模式。它属于机器学习的分支，尤其在**深度学习**领域广泛应用，能够处理图像、语音、文本等非结构化数据。
    
    ### 2. **核心结构**
    - **分层设计**：典型的神经网络包括：
      - **输入层**：接收原始数据（如图像像素、文本词向量）。
      - **隐藏层**（多层）：通过数学运算提取数据的抽象特征。层数越多，网络越“深”（即深度学习）。
      - **输出层**：生成最终结果（如分类标签、预测值）。
    - **神经元**：每个神经元接收前一层输出的加权输入，通过**激活函数**（如ReLU、Sigmoid）引入非线性，决定是否传递信号。
    
    ### 3. **工作原理**
    - **前向传播**：数据从输入层逐层传递，每层神经元计算加权和并应用激活函数，最终输出预测结果。
    - **损失函数**：比较预测结果与真实值之间的误差（如交叉熵、均方误差）。
    - **反向传播**：通过梯度下降算法，从输出层反向调整各层神经元的**权重**和**偏置**，以最小化误差。这一过程依赖链式求导法则。
    - **训练迭代**：重复前向传播和反向传播，直到模型收敛（误差足够小或达到训练轮次）。
    
    ### 4. **神经网络的类型**
    - **卷积神经网络（CNN）**：专为图像设计，使用卷积核自动学习局部特征（如边缘、纹理），广泛应用于物体检测、医学影像分析。
    - **循环神经网络（RNN）**：处理序列数据（如文本、时间序列），通过反馈连接保留上下文信息，适用于语言翻译、语音识别。
    - **其他变体**：如长短期记忆网络（LSTM）解决RNN的梯度消失问题，Transformer模型（如GPT）通过自注意力机制处理长序列依赖。
    
    ### 5. **关键优势**
    - **自动特征提取**：无需人工设计特征，直接从数据中学习层次化表示。
    - **大规模数据处理**：依赖计算能力和海量数据（如ImageNet、语料库）提升性能。
    - **多领域应用**：从计算机视觉到自然语言处理（NLP），推动自动驾驶、医疗诊断等突破。
    
    ### 6. **局限性**
    - **数据依赖**：需要大量标注数据，否则易过拟合。
    - **计算资源**：训练深度网络需高性能GPU/TPU。
    - **黑箱问题**：决策过程难以解释（推动可解释AI/XAI的发展）。
    
    ### 示例应用（来自上下文）
    - **图像识别**：CNN在面部识别中分析像素层次特征。
    - **自然语言处理**：RNN或Transformer生成连贯文本。
    - **医疗诊断**：深度学习模型从X光片中检测病变。
    
    总结来说，神经网络通过模拟人脑的互联学习机制，利用多层非线性变换从数据中挖掘深层规律，成为现代人工智能的核心技术之一。
    
    === 第二轮：含反馈 ===
    从PDF中提取文本...
    对文本进行分块...
    创建了 13 个文本块
    为文本块创建嵌入...
    已将 13 个块添加到向量存储中
    使用高质量反馈微调索引...
    已添加来自反馈的增强内容: 什么是神经网络以及它如何工作？...
    微调后的索引现在有 14 个项目 (原始: 13)
    
    查询 1: 什么是神经网络以及它如何工作？
    
    === 使用反馈增强型RAG处理查询 ===
    查询: 什么是神经网络以及它如何工作？
    基于反馈历史调整相关性分数...
      文档 1: 基于 1 条反馈，分数从 0.8559 调整为 1.1127
      文档 2: 基于 1 条反馈，分数从 0.7801 调整为 1.0141
      文档 3: 基于 1 条反馈，分数从 0.7700 调整为 1.0010
    正在生成响应...
    
    === 响应 ===
    根据上下文，神经网络（尤其是人工神经网络）是深度学习的核心组成部分，其设计灵感来源于人脑的结构和功能。以下是关于神经网络及其工作原理的详细说明：
    
    ### 1. **神经网络的定义**
    神经网络是一种由多层互联的“神经元”（或节点）组成的计算模型，用于分析和学习数据中的复杂模式。它属于机器学习的分支，尤其在**深度学习**领域广泛应用，能够处理图像、语音、文本等非结构化数据。
    
    ### 2. **核心结构**
    - **分层设计**：典型的神经网络包括：
      - **输入层**：接收原始数据（如图像像素、文本词向量）。
      - **隐藏层**（多层）：通过数学运算提取数据的抽象特征。层数越多，网络越“深”（即深度学习）。
      - **输出层**：生成最终结果（如分类标签、预测值）。
    - **神经元**：每个神经元接收前一层输出的加权输入，通过**激活函数**（如ReLU、Sigmoid）引入非线性，决定是否传递信号。
    
    ### 3. **工作原理**
    - **前向传播**：数据从输入层逐层传递，每层神经元计算加权和并应用激活函数，最终输出预测结果。
    - **损失函数**：比较预测结果与真实值之间的误差（如交叉熵、均方误差）。
    - **反向传播**：通过梯度下降算法，从输出层反向调整各层神经元的**权重**和**偏置**，以最小化误差。这一过程依赖链式求导法则。
    - **训练迭代**：重复前向传播和反向传播，直到模型收敛（误差足够小或达到训练轮次）。
    
    ### 4. **神经网络的类型**
    - **卷积神经网络（CNN）**：专为图像设计，使用卷积核自动学习局部特征（如边缘、纹理），广泛应用于物体检测、医学影像分析。
    - **循环神经网络（RNN）**：处理序列数据（如文本、时间序列），通过反馈连接保留上下文信息，适用于语言翻译、语音识别。
    - **其他变体**：如长短期记忆网络（LSTM）解决RNN的梯度消失问题，Transformer模型（如GPT）通过自注意力机制处理长序列依赖。
    
    ### 5. **关键优势**
    - **自动特征提取**：无需人工设计特征，直接从数据中学习层次化表示。
    - **大规模数据处理**：依赖计算能力和海量数据（如ImageNet、语料库）提升性能。
    - **多领域应用**：从计算机视觉到自然语言处理（NLP），推动自动驾驶、医疗诊断等突破。
    
    ### 6. **局限性**
    - **数据依赖**：需要大量标注数据，否则易过拟合。
    - **计算资源**：训练深度网络需高性能GPU/TPU。
    - **黑箱问题**：决策过程难以解释（推动可解释AI/XAI的发展）。
    
    ### 示例应用（来自上下文）
    - **图像识别**：CNN在面部识别中分析像素层次特征。
    - **自然语言处理**：RNN或Transformer生成连贯文本。
    - **医疗诊断**：深度学习模型从X光片中检测病变。
    
    总结来说，神经网络通过模拟人脑的互联学习机制，利用多层非线性变换从数据中挖掘深层规律，成为现代人工智能的核心技术之一。
    
    === 正在比较结果 ===
    
    查询 1: 什么是神经网络以及它如何工作？
    分析: ### 对比分析：标准RAG系统 vs. 反馈增强型RAG
    
    #### 1. **与查询的相关性**
    - **标准RAG系统**：响应高度相关，全面覆盖了神经网络的定义、结构、工作原理、类型、优势、局限性和应用示例，完全匹配用户查询需求。
    - **反馈增强型RAG**：响应内容与标准版本几乎完全一致，未体现额外的相关性优化。两者均能精准回答用户问题，但反馈版本未展示出相关性上的改进。
    
    **结论*...



```python
########################################
# # Run a full RAG workflow
########################################

# Run an interactive example
print("\n\n=== INTERACTIVE EXAMPLE ===")
print("Enter your query about AI:")
user_query = input()

# Load accumulated feedback
all_feedback = load_feedback_data()

# Run full workflow
result = full_rag_workflow(
    pdf_path=pdf_path,
    query=user_query,
    feedback_data=all_feedback,
    fine_tune=True
)

########################################
# # Run a full RAG workflow
########################################
```

    
    
    === INTERACTIVE EXAMPLE ===
    Enter your query about AI:
    未找到反馈数据文件。将以空反馈开始。
    从PDF中提取文本...
    对文本进行分块...
    创建了 13 个文本块
    为文本块创建嵌入...
    已将 13 个块添加到向量存储中
    
    === 使用反馈增强型RAG处理查询 ===
    查询: 什么是神经网络以及它如何工作？
    正在生成响应...
    
    === 响应 ===
    根据上下文内容，神经网络（尤其是**人工神经网络**）是受生物大脑结构和功能启发而设计的计算模型，属于深度学习的核心组成部分。以下是其定义和工作原理的详细说明：
    
    ---
    
    ### **1. 神经网络的定义**
    - **基本概念**：神经网络是由多层互连的“神经元”（节点）组成的系统，用于分析和学习数据中的复杂模式。它通过模拟人脑神经元之间的信号传递来处理信息。
    - **核心特点**：
      - **分层结构**：通常包含输入层、隐藏层（可能有多层）和输出层。
      - **自适应学习**：通过调整神经元之间的连接权重（参数）来优化模型性能。
      - **非线性处理**：激活函数（如ReLU、Sigmoid）引入非线性，使网络能拟合复杂数据。
    
    ---
    
    ### **2. 神经网络的工作原理**
    #### **（1）数据输入与前向传播**
    - **输入层**：接收原始数据（如图像像素、文本向量）。
    - **隐藏层**：每层神经元对输入进行加权求和，并通过激活函数生成输出。  
      - **示例**：在图像识别中，底层可能检测边缘，高层组合边缘形成物体轮廓。
    - **输出层**：生成最终预测（如分类标签、数值）。
    
    #### **（2）损失计算与反向传播**
    - **损失函数**：衡量预测值与真实值的误差（如交叉熵、均方误差）。
    - **反向传播**：  
      - 从输出层反向计算误差对每个权重的梯度。  
      - 使用优化算法（如梯度下降）调整权重，逐步最小化误差。
    
    #### **（3）训练与优化**
    - **迭代学习**：重复前向传播和反向传播，直到模型收敛。
    - **正则化技术**：防止过拟合（如Dropout、L2正则化）。
    
    ---
    
    ### **3. 神经网络的类型（基于上下文）**
    - **卷积神经网络（CNN）**：  
      - **用途**：处理图像/视频数据，通过卷积核自动提取局部特征（如边缘、纹理）。  
      - **应用**：医学图像分析、人脸识别（上下文提到的“物体检测”）。
    - **循环神经网络（RNN）**：  
      - **用途**：处理序列数据（文本、时间序列），通过反馈连接记忆历史信息。  
      - **应用**：语言翻译、语音识别（上下文提到的“情感分析”）。
    
    ---
    
    ### **4. 神经网络的训练数据依赖**
    - **监督学习**：依赖标记数据（如图像分类中的“猫/狗”标签）。  
    - **无监督学习**：从无标记数据中发现模式（如聚类用户行为）。
    
    ---
    
    ### **5. 神经网络的突破性应用（基于上下文）**
    - **自然语言处理（NLP）**：如聊天机器人、文本摘要。  
    - **计算机视觉**：如自动驾驶中的环境感知（上下文提到的“监控系统”）。  
    - **医疗保健**：分析医学影像、辅助诊断（上下文提到的“预测患者预后”）。
    
    ---
    
    ### **6. 神经网络的挑战**
    - **黑箱问题**：决策过程难以解释（上下文提到“透明度和可解释性”）。  
    - **数据偏见**：训练数据偏差可能导致模型歧视（如种族或性别偏见）。  
    - **计算资源**：深度网络需要大量数据和算力。
    
    ---
    
    ### **总结**
    神经网络通过模拟生物神经元的交互，实现了从数据中自动学习复杂模式的能力。其核心在于分层特征提取和权重优化，广泛应用于图像、文本、医疗等领域。然而，伦理问题（如偏见）和技术限制（如可解释性）仍需持续研究（与上下文中“伦理影响”和“XAI可解释人工智能”相呼应）。
    
    === 您是否愿意对这个响应提供反馈？ ===
    评分相关性（1-5，5表示最相关）：
    评分质量（1-5，5表示最高质量）：
    有任何评论吗？（可选，按Enter跳过）
    反馈已记录。感谢您的参与！


# 10.9 可视化评估结果


```python
# 提取包含反馈影响分析的比较数据
comparisons = evaluation_results['comparison']

# 打印分析结果以可视化反馈影响
print("\n=== 反馈影响分析 ===\n")
for i, comparison in enumerate(comparisons):
    print(f"查询 {i+1}: {comparison['query']}")
    print(f"\n反馈影响分析:")
    print(comparison['analysis'])
    print("\n" + "-"*50 + "\n")

# 此外，我们可以比较各轮之间的某些指标
round_responses = [evaluation_results[f'round{round_num}_results'] for round_num in range(1, len(evaluation_results) - 1)]
response_lengths = [[len(r["response"]) for r in round] for round in round_responses]

# 比较响应长度（作为完整性的代理）
print("\n响应长度比较（完整性代理）：")
avg_lengths = [sum(lengths) / len(lengths) for lengths in response_lengths]
for round_num, avg_len in enumerate(avg_lengths, start=1):
    print(f"轮次 {round_num}: {avg_len:.1f} 字符")

if len(avg_lengths) > 1:
    # 计算每轮之间的变化百分比
    changes = [(avg_lengths[i] - avg_lengths[i-1]) / avg_lengths[i-1] * 100 for i in range(1, len(avg_lengths))]
    for round_num, change in enumerate(changes, start=2):
        print(f"从轮次 {round_num-1} 到 轮次 {round_num} 的变化: {change:.1f}%")

```

    
    === 反馈影响分析 ===
    
    查询 1: 什么是神经网络以及它如何工作？
    
    反馈影响分析:
    ### 对比分析：标准RAG系统 vs. 反馈增强型RAG
    
    #### 1. **与查询的相关性**
    - **标准RAG系统**：响应高度相关，全面覆盖了神经网络的定义、结构、工作原理、类型、优势、局限性和应用示例，完全匹配用户查询需求。
    - **反馈增强型RAG**：响应内容与标准版本几乎完全一致，未体现额外的相关性优化。两者均能精准回答用户问题，但反馈版本未展示出相关性上的改进。
    
    **结论**：两者相关性表现相当，反馈机制未在此维度体现提升。
    
    #### 2. **信息准确性**
    - **标准RAG系统**：所有技术细节（如前向传播、反向传播、激活函数等）均准确无误，与参考答案和领域知识一致。
    - **反馈增强型RAG**：内容准确性同样优秀，但未修正标准版本中可能的潜在错误（如某些术语的简化表述），也未补充更权威的来源或数据。
    
    **结论**：两者准确性相当，反馈机制未显著提升纠错能力。
    
    #### 3. **回答完整性**
    - **标准RAG系统**：回答非常完整，从基础定义到实际应用均涵盖，且逻辑清晰、层次分明。
    - **反馈增强型RAG**：内容完整性与标准版本完全相同，未补充额外信息（如最新研究进展或用户可能关注的衍生问题）。
    
    **结论**：两者完整性一致，反馈机制未扩展回答的广度或深度。
    
    #### 4. **表述清晰度与简洁性**
    - **标准RAG系统**：表述清晰，但部分技术细节（如反向传播的数学原理）可能对初学者稍显复杂。结构虽合理，但篇幅较长。
    - **反馈增强型RAG**：未优化表述方式，未简化复杂概念或调整结构以提升可读性。与标准版本相比，未体现更简洁或更易懂的改进。
    
    **结论**：两者表述水平相同，反馈机制未优化语言表达。
    
    ---
    
    ### **关键发现与改进建议**
    1. **反馈机制未体现改进的原因**：
       - **内容重复性**：反馈增强型RAG的响应与标准版本完全一致，可能因反馈循环未实际触发（如无用户历史数据或未检测到优化点）。
       - **潜在优化方向**：若反馈机制能结合用户交互（如追问“能否用更简单的例子说明反向传播？”），可针对性简化表述或补充示例。
    
    2. **标准RAG的潜在优势**：
       - 在无用户反馈数据时，标准版本已能生成高质量回答，说明基础检索与生成能力较强。
    
    3. **反馈机制应有的价值**（未在此案例中体现）：
       - **动态调整**：根据用户历史查询优化回答重点（如更强调应用场景）。
       - **错误修正**：若标准版本存在错误，反馈机制应优先修正。
       - **个性化**：适应不同用户的知识水平（如初学者 vs. 研究者）。
    
    ---
    
    ### **最终结论**
    - **当前表现**：两个版本响应质量**完全相同**，反馈增强型RAG未展示出任何改进。
    - **原因分析**：可能是测试查询未触发反馈优化，或反馈循环未有效集成到生成流程中。
    - **改进建议**：
      - 确保反馈机制能识别用户意图差异（如技术深度需求），动态调整回答详略。
      - 引入用户交互历史，优化后续回答的针对性（如补充“您是否想了解具体数学推导？”）。
      - 对标准版本中可能的模糊表述进行迭代优化（如用比喻解释激活函数）。
    
    --------------------------------------------------
    
    
    响应长度比较（完整性代理）：
    轮次 1: 1285.0 字符

