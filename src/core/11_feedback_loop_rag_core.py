"""
反馈循环机制(Feedback Loop) 核心函数
"""
import os
import json
import fitz
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
llm_model = os.getenv("LLM_MODEL_ID")
embedding_model = os.getenv("EMBEDDING_MODEL_ID")


####################################
# 反馈循环机制的RAG完整流程
####################################
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


####################################
# 完整的RAG
####################################
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

    print("\n=== 回答 ===")
    print(response)

    return result


####################################
# 调用LLM评估每个历史反馈项的相关性
####################################
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


####################################
# 根据反馈调整检索到的文档的相关性分数
####################################
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
            result["similarity"] = adjusted_score  # 更新主分数
            result["relevance_score"] = adjusted_score  # 更新相关性分数
            result["feedback_applied"] = True  # 标记反馈已应用
            result["feedback_count"] = len(relevant_feedback)  # 使用的反馈条目数量

            # 记录调整细节
            print(
                f"  文档 {i + 1}: 基于 {len(relevant_feedback)} 条反馈，分数从 {original_score:.4f} 调整为 {adjusted_score:.4f}")

    # 按调整后的分数重新排序结果，确保更高匹配质量的结果优先显示
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results


####################################
# 通过结合高质量的过往交互微调向量索引
#   这将从成功的问答对中创建增强的可检索内容
####################################
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
                "query": feedback["query"],  # 保存原始查询以供参考
                "relevance_score": 1.2,  # 提高初始相关性以优先处理这些项
                "feedback_count": 1,  # 跟踪反馈整合情况
                "original_feedback": feedback  # 保存完整的反馈记录
            }
        )

        print(f"已添加来自反馈的增强内容: {feedback['query'][:50]}...")

    # 记录关于增强的汇总统计信息
    print(f"微调后的索引现在有 {len(new_store.texts)} 个项目 (原始: {len(chunks)})")
    return new_store


####################################
# 反馈检索组件函数
####################################
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


#####
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
                "index": i,  # 在原始文档中的位置
                "source": pdf_path,  # 源文档路径
                "relevance_score": 1.0,  # 初始相关性分数（将通过反馈更新）
                "feedback_count": 0  # 接收到此块反馈的计数器
            }
        )

    print(f"已将 {len(chunks)} 个块添加到向量存储中")
    return chunks, store


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
