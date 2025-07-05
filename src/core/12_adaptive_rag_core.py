"""
自适应检索 核心函数
"""
import os
import re
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
# 自适应检索RAG完整流程
####################################
def rag_with_adaptive_retrieval(pdf_path, query, k=4, user_context=None):
    """
    完整的RAG管道，带有自适应检索功能。

    Args:
        pdf_path (str): PDF文档的路径
        query (str): 用户查询
        k (int): 要检索的文档数量
        user_context (str): 可选的用户上下文

    Returns:
        Dict: 包含查询、检索到的文档、查询类型和响应的结果字典
    """
    print("\n=== RAG WITH ADAPTIVE RETRIEVAL ===")
    print(f"Query: {query}")  # 打印查询内容

    # 处理文档以提取文本，将其分块，并创建嵌入向量
    chunks, vector_store = process_document(pdf_path)

    # 对查询进行分类以确定其类型
    query_type = classify_query(query)
    print(f"Query classified as: {query_type}")  # 打印查询被分类为的类型

    # 根据查询类型使用自适应检索策略检索文档
    retrieved_docs = adaptive_retrieval(query, vector_store, k, user_context)

    # 根据查询、检索到的文档和查询类型生成响应
    response = generate_response(query, retrieved_docs, query_type)

    # 将结果编译成一个字典
    result = {
        "query": query,  # 用户查询
        "query_type": query_type,  # 查询类型
        "retrieved_documents": retrieved_docs,  # 检索到的文档
        "response": response  # 生成的响应
    }

    print("\n=== RESPONSE ===")  # 打印响应标题
    print(response)  # 打印生成的响应

    return result  # 返回结果字典


####################################
# 查询分类，执行自适应检索
####################################
def adaptive_retrieval(query, vector_store, k=4, user_context=None):
    """
    执行自适应检索，通过选择并执行适当的检索策略。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 要检索的文档数量
        user_context (str): 可选的用户上下文，用于上下文相关的查询

    Returns:
        List[Dict]: 检索到的文档列表
    """
    # 对查询进行分类以确定其类型
    query_type = classify_query(query)
    print(f"查询被分类为: {query_type}")

    # 根据查询类型选择并执行适当的检索策略
    if query_type == "Factual":
        # 使用事实检索策略获取精确信息
        results = factual_retrieval_strategy(query, vector_store, k)
    elif query_type == "Analytical":
        # 使用分析检索策略实现全面覆盖
        results = analytical_retrieval_strategy(query, vector_store, k)
    elif query_type == "Opinion":
        # 使用观点检索策略获取多样化的观点
        results = opinion_retrieval_strategy(query, vector_store, k)
    elif query_type == "Contextual":
        # 使用上下文检索策略，并结合用户上下文
        results = contextual_retrieval_strategy(query, vector_store, k, user_context)
    else:
        # 如果分类失败，默认使用事实检索策略
        results = factual_retrieval_strategy(query, vector_store, k)

    return results  # 返回检索到的文档


####################################
# 检索策略实现方案
#   1. factual_retrieval_strategy：事实性查询的检索策略，专注于精确度
#   2. analytical_retrieval_strategy：析性查询的检索策略，专注于全面覆盖
#   3. opinion_retrieval_strategy：视点性查询的检索策略，专注于多角度思考和观点表达
#   4. contextual_retrieval_strategy：上下文检索策略，结合用户上下文信息
####################################
def factual_retrieval_strategy(query, vector_store, k=4):
    """
    针对事实性查询的检索策略，专注于精确度。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储库
        k (int): 返回的文档数量

    Returns:
        List[Dict]: 检索到的文档列表
    """
    print(f"执行事实性检索策略: '{query}'")

    # 使用LLM增强查询以提高精确度
    system_prompt = """您是搜索查询优化专家。
        您的任务是重构给定的事实性查询，使其更精确具体以提升信息检索效果。
        重点关注关键实体及其关联关系。

        请仅提供优化后的查询，不要包含任何解释。
    """

    user_prompt = f"请优化此事实性查询: {query}"

    # 使用LLM生成增强后的查询
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 提取并打印增强后的查询
    enhanced_query = response.choices[0].message.content.strip()
    print(f"优化后的查询: {enhanced_query}")

    # 为增强后的查询创建嵌入向量
    query_embedding = create_embeddings(enhanced_query)

    # 执行初始相似性搜索以检索文档
    initial_results = vector_store.similarity_search(query_embedding, k=k * 2)

    # 初始化一个列表来存储排序后的结果
    ranked_results = []

    # 使用LLM对文档进行评分和排序
    for doc in initial_results:
        relevance_score = score_document_relevance(enhanced_query, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "relevance_score": relevance_score
        })

    # 按相关性得分降序排列结果
    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)

    # 返回前k个结果
    return ranked_results[:k]


def analytical_retrieval_strategy(query, vector_store, k=4):
    """
    针对分析性查询的检索策略，专注于全面覆盖。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储库
        k (int): 返回的文档数量

    Returns:
        List[Dict]: 检索到的文档列表
    """
    print(f"执行分析性检索策略: '{query}'")

    # 定义系统提示以指导AI生成子问题
    system_prompt = """您是复杂问题拆解专家。
    请针对给定的分析性查询生成探索不同维度的子问题。
    这些子问题应覆盖主题的广度并帮助获取全面信息。

    请严格生成恰好3个子问题，每个问题单独一行。
    """

    # 创建包含主查询的用户提示
    user_prompt = f"请为此分析性查询生成子问题：{query}"

    # 使用LLM生成子问题
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    # 提取并清理子问题
    sub_queries = response.choices[0].message.content.strip().split('\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]
    print(f"生成的子问题: {sub_queries}")

    # 为每个子问题检索文档
    all_results = []
    for sub_query in sub_queries:
        # 为子问题创建嵌入向量
        sub_query_embedding = create_embeddings(sub_query)
        # 执行相似性搜索以获取子问题的结果
        results = vector_store.similarity_search(sub_query_embedding, k=2)
        all_results.extend(results)

    # 确保多样性，从不同的子问题结果中选择
    # 移除重复项（相同的文本内容）
    unique_texts = set()
    diverse_results = []

    for result in all_results:
        if result["text"] not in unique_texts:
            unique_texts.add(result["text"])
            diverse_results.append(result)

    # 如果需要更多结果以达到k，则从初始结果中添加更多
    if len(diverse_results) < k:
        # 对主查询直接检索
        main_query_embedding = create_embeddings(query)
        main_results = vector_store.similarity_search(main_query_embedding, k=k)

        for result in main_results:
            if result["text"] not in unique_texts and len(diverse_results) < k:
                unique_texts.add(result["text"])
                diverse_results.append(result)

    # 返回前k个多样化的结果
    return diverse_results[:k]


def opinion_retrieval_strategy(query, vector_store, k=4):
    """
    针对观点查询的检索策略，专注于多样化的观点。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储库
        k (int): 返回的文档数量

    Returns:
        List[Dict]: 检索到的文档列表
    """
    print(f"执行观点检索策略: '{query}'")

    # 定义系统提示以指导AI识别不同观点
    system_prompt = """您是主题多视角分析专家。
        针对给定的观点类或意见类查询，请识别人们可能持有的不同立场或观点。

        请严格返回恰好3个不同观点角度，每个角度单独一行。
    """

    # 创建包含主查询的用户提示
    user_prompt = f"请识别以下主题的不同观点：{query}"

    # 使用LLM生成不同的观点
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    # 提取并清理观点
    viewpoints = response.choices[0].message.content.strip().split('\n')
    viewpoints = [v.strip() for v in viewpoints if v.strip()]
    print(f"已识别的观点: {viewpoints}")

    # 检索代表每个观点的文档
    all_results = []
    for viewpoint in viewpoints:
        # 将主查询与观点结合
        combined_query = f"{query} {viewpoint}"
        # 为组合查询创建嵌入向量
        viewpoint_embedding = create_embeddings(combined_query)
        # 执行相似性搜索以获取组合查询的结果
        results = vector_store.similarity_search(viewpoint_embedding, k=2)

        # 标记结果所代表的观点
        for result in results:
            result["viewpoint"] = viewpoint

        # 将结果添加到所有结果列表中
        all_results.extend(results)

    # 选择多样化的意见范围
    # 尽量确保从每个观点中至少获得一个文档
    selected_results = []
    for viewpoint in viewpoints:
        # 按观点过滤文档
        viewpoint_docs = [r for r in all_results if r.get("viewpoint") == viewpoint]
        if viewpoint_docs:
            selected_results.append(viewpoint_docs[0])

    # 用最高相似度的文档填充剩余的槽位
    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        # 按相似度排序剩余文档
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["similarity"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])

    # 返回前k个结果
    return selected_results[:k]


def contextual_retrieval_strategy(query, vector_store, k=4, user_context=None):
    """
    针对上下文查询的检索策略，结合用户提供的上下文信息。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储库
        k (int): 返回的文档数量
        user_context (str): 额外的用户上下文信息

    Returns:
        List[Dict]: 检索到的文档列表
    """
    print(f"执行上下文检索策略: '{query}'")

    # 如果未提供用户上下文，则尝试从查询中推断上下文
    if not user_context:
        system_prompt = """您是理解查询隐含上下文的专家。
        对于给定的查询，请推断可能相关或隐含但未明确说明的上下文信息。
        重点关注有助于回答该查询的背景信息。

        请简要描述推断的隐含上下文。
        """

        user_prompt = f"推断此查询中的隐含背景(上下文)：{query}"

        # 使用LLM生成推断出的上下文
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        # 提取并打印推断出的上下文
        user_context = response.choices[0].message.content.strip()
        print(f"推断出的上下文: {user_context}")

    # 重新表述查询以结合上下文
    system_prompt = """您是上下文整合式查询重构专家。
    根据提供的查询和上下文信息，请重新构建更具体的查询以整合上下文，从而获取更相关的信息。

    请仅返回重新构建的查询，不要包含任何解释。
    """

    user_prompt = f"""
    原始查询：{query}
    关联上下文：{user_context}

    请结合此上下文重新构建查询：
    """

    # 使用LLM生成结合上下文的查询
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 提取并打印结合上下文的查询
    contextualized_query = response.choices[0].message.content.strip()
    print(f"结合上下文的查询: {contextualized_query}")

    # 基于结合上下文的查询检索文档
    query_embedding = create_embeddings(contextualized_query)
    initial_results = vector_store.similarity_search(query_embedding, k=k * 2)

    # 根据相关性和用户上下文对文档进行排序
    ranked_results = []

    for doc in initial_results:
        # 计算文档在考虑上下文情况下的相关性得分
        context_relevance = score_document_context_relevance(query, user_context, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "context_relevance": context_relevance
        })

    # 按上下文相关性排序，并返回前k个结果
    ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
    return ranked_results[:k]


####

####################################
# 文档评分组件函数
####################################
def score_document_relevance(query, document):
    """
    使用LLM对文档与查询的相关性进行评分。

    Args:
        query (str): 用户查询
        document (str): 文档文本

    Returns:
        float: 相关性评分，范围为0-10
    """
    # 系统提示，指导模型如何评估相关性
    system_prompt = """您是文档相关性评估专家。
        请根据文档与查询的匹配程度给出0到10分的评分：
        0 = 完全无关
        10 = 完美契合查询

        请仅返回一个0到10之间的数字评分，不要包含任何其他内容。
    """

    # 如果文档过长，则截断文档
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document

    # 包含查询和文档预览的用户提示
    user_prompt = f"""
        查询: {query}

        文档: {doc_preview}

        相关性评分（0-10）：
    """

    # 使用模型生成响应
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从模型的响应中提取评分
    score_text = response.choices[0].message.content.strip()

    # 使用正则表达式提取数值评分
    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # 确保评分在0-10范围内
    else:
        # 如果提取失败，则返回默认评分
        return 5.0


def score_document_context_relevance(query, context, document):
    """
    根据查询和上下文评估文档的相关性。

    Args:
        query (str): 用户查询
        context (str): 用户上下文
        document (str): 文档文本

    Returns:
        float: 相关性评分，范围为0-10
    """
    # 系统提示，指导模型如何根据上下文评估相关性
    system_prompt = """您是结合上下文评估文档相关性的专家。
        请根据文档在给定上下文中对查询的响应质量，给出0到10分的评分：
        0 = 完全无关
        10 = 在给定上下文中完美契合查询

        请严格仅返回一个0到10之间的数字评分，不要包含任何其他内容。
    """

    # 如果文档过长，则截断文档
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document

    # 包含查询、上下文和文档预览的用户提示
    user_prompt = f"""
    待评估查询：{query}
    关联上下文：{context}

    文档内容预览：
    {doc_preview}

    结合上下文的相关性评分（0-10）：
    """

    # 使用模型生成响应
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从模型的响应中提取评分
    score_text = response.choices[0].message.content.strip()

    # 使用正则表达式提取数值评分
    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # 确保评分在0-10范围内
    else:
        # 如果提取失败，则返回默认评分
        return 5.0


####

####################################
# 查询分类
####################################
def classify_query(query):
    """
    将查询分类为四个类别之一：事实性（Factual）、分析性（Analytical）、意见性（Opinion）或上下文相关性（Contextual）。

    Args:
        query (str): 用户查询

    Returns:
        str: 查询类别
    """
    # 定义系统提示以指导AI进行分类
    system_prompt = """您是专业的查询分类专家。
        请将给定查询严格分类至以下四类中的唯一一项：
        - Factual：需要具体、可验证信息的查询
        - Analytical：需要综合分析或深入解释的查询
        - Opinion：涉及主观问题或寻求多元观点的查询
        - Contextual：依赖用户具体情境的查询

        请仅返回分类名称，不要添加任何解释或额外文本。
    """

    # 创建包含要分类查询的用户提示
    user_prompt = f"对以下查询进行分类: {query}"

    # 从AI模型生成分类响应
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从响应中提取并去除多余的空白字符以获取类别
    category = response.choices[0].message.content.strip()

    # 定义有效的类别列表
    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]

    # 确保返回的类别是有效的
    for valid in valid_categories:
        if valid in category:
            return valid

    # 如果分类失败，默认返回“Factual”（事实性）
    return "Factual"


def generate_response(query, results, query_type):
    """
    根据查询、检索到的文档和查询类型生成响应。

    Args:
        query (str): 用户查询
        results (List[Dict]): 检索到的文档列表
        query_type (str): 查询类型

    Returns:
        str: 生成的响应
    """
    # 从检索到的文档中准备上下文，通过连接它们的文本并使用分隔符
    context = "\n\n---\n\n".join([r["text"] for r in results])

    # 根据查询类型创建自定义系统提示
    if query_type == "Factual":
        system_prompt = """您是基于事实信息应答的AI助手。
    请严格根据提供的上下文回答问题，确保信息准确无误。
    若上下文缺乏必要信息，请明确指出信息局限。"""

    elif query_type == "Analytical":
        system_prompt = """您是专业分析型AI助手。
    请基于提供的上下文，对主题进行多维度深度解析：
    - 涵盖不同层面的关键要素（不同方面和视角）
    - 整合多方观点形成系统分析
    若上下文存在信息缺口或空白，请在分析时明确指出信息短缺。"""

    elif query_type == "Opinion":
        system_prompt = """您是观点整合型AI助手。
    请基于提供的上下文，结合以下标准给出不同观点：
    - 全面呈现不同立场观点
    - 保持各观点表述的中立平衡，避免出现偏见
    - 当上下文视角有限时，直接说明"""

    elif query_type == "Contextual":
        system_prompt = """您是情境上下文感知型AI助手。
    请结合查询背景与上下文信息：
    - 建立问题情境与文档内容的关联
    - 当上下文无法完全匹配具体情境时，请明确说明适配性限制"""

    else:
        system_prompt = """您是通用型AI助手。请基于上下文回答问题，若信息不足请明确说明。"""

    # 通过结合上下文和查询创建用户提示
    user_prompt = f"""
    上下文:
    {context}

    问题: {query}

    请基于上下文提供专业可靠的回答。
    """

    # 使用 OpenAI 生成响应
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    # 返回生成的响应内容
    return response.choices[0].message.content


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    为RAG处理文档。

    Args:
    pdf_path (str): PDF文件的路径。
    chunk_size (int): 每个文本块的大小（以字符为单位）。
    chunk_overlap (int): 文本块之间的重叠大小（以字符为单位）。

    Returns:
    Tuple[List[str], SimpleVectorStore]: 包含文档文本块及其嵌入向量的向量存储。
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
    return chunks, store


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
