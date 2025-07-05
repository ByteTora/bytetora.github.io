"""
重排（LLM重排、关键词重排） 核心函数
"""
import os
import re
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
# 重排的完整流程
####################################
def rag_with_reranking(query, vector_store, reranking_method="llm", top_n=3):
    """
    完整的RAG管道，包含重排序功能。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        reranking_method (str): 重排序方法 ('llm' 或 'keywords')
        top_n (int): 重排序后返回的结果数量

    Returns:
        Dict: 包括查询、上下文和响应的结果
    """
    # 创建查询嵌入
    query_embedding = create_embeddings(query)

    # 初步检索（获取比重排序所需更多的结果）
    initial_results = vector_store.similarity_search(query_embedding, k=10)

    # 应用重排序
    if reranking_method == "llm":
        reranked_results = rerank_with_llm(query, initial_results, top_n=top_n)
    elif reranking_method == "keywords":
        reranked_results = rerank_with_keywords(query, initial_results, top_n=top_n)
    else:
        # 不进行重排序，直接使用初步检索的前几个结果
        reranked_results = initial_results[:top_n]

    # 将重排序结果的上下文合并
    context = "\n\n===\n\n".join([result["text"] for result in reranked_results])

    # 根据上下文生成响应
    response = generate_response(query, context)

    return {
        "query": query,
        "reranking_method": reranking_method,
        "initial_results": initial_results[:top_n],
        "reranked_results": reranked_results,
        "context": context,
        "response": response
    }


####################################
# 基于 LLM 的重排序
####################################
def rerank_with_llm(query, results, top_n=3):
    """
    使用 LLM 相关性评分对搜索结果进行重排序。

    Args:
        query (str): 用户查询
        results (List[Dict]): 初始搜索结果
        top_n (int): 重排序后要返回的结果数量

    Returns:
        List[Dict]: 重排序后的结果
    """
    print(f"正在重排序 {len(results)} 个文档...")  # 打印要重排序的文档数量

    scored_results = []  # 初始化一个空列表以存储评分结果

    # 定义 LLM 的系统提示
    system_prompt = """
    您是文档相关性评估专家，擅长判断文档与搜索查询的匹配程度。您的任务是根据文档对给定查询的应答质量，给出0到10分的评分。

    评分标准：
    0-2分：文档完全无关
    3-5分：文档含部分相关信息但未直接回答问题
    6-8分：文档相关且能部分解答查询
    9-10分：文档高度相关且直接准确回答问题

    必须仅返回0到10之间的单个整数评分，不要包含任何其他内容。
    """

    # 遍历每个结果
    for i, result in enumerate(results):
        # 每 5 个文档显示一次进度
        if i % 5 == 0:
            print(f"正在评分文档 {i + 1}/{len(results)}...")

        # 定义 LLM 的用户提示
        user_prompt = f"""
        查询: {query}

        文档:
        {result['text']}

        请对文档的相关性进行评分，评分范围为 0 到 10, 并仅返回一个整数。
        """

        # 获取 LLM 的响应
        response = client.chat.completions.create(
            model=llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # 从 LLM 响应中提取评分
        score_text = response.choices[0].message.content.strip()

        # 使用正则表达式提取数值评分
        score_match = re.search(r'\b(10|[0-9])\b', score_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            # 如果评分提取失败，使用相似度评分作为备选
            print(f"警告：无法从响应中提取评分：'{score_text}'，使用相似度评分代替")
            score = result["similarity"] * 10

        # 将评分结果添加到列表中
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": score
        })

    # 按相关性评分降序对结果进行排序
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)

    # 返回前 top_n 个结果
    return reranked_results[:top_n]


####################################
# 基于关键词的重排序
####################################
def rerank_with_keywords(query, results, top_n=3):
    """
    基于关键词匹配次数和位置的简单重排序方法。

    Args:
        query (str): 用户查询
        results (List[Dict]): 初始搜索结果
        top_n (int): 重排序后返回的结果数量

    Returns:
        List[Dict]: 重排序后的结果
    """
    # 从查询中提取重要关键词
    keywords = [word.lower() for word in query.split() if len(word) > 3]

    scored_results = []  # 初始化一个列表以存储评分结果

    for result in results:
        document_text = result["text"].lower()  # 将文档文本转换为小写

        # 基础分数从向量相似度开始
        base_score = result["similarity"] * 0.5

        # 初始化关键词分数
        keyword_score = 0
        for keyword in keywords:
            if keyword in document_text:
                # 每找到一个关键词加一些分数
                keyword_score += 0.1

                # 如果关键词出现在文本开头部分，额外加分
                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:  # 在文本的前四分之一部分
                    keyword_score += 0.1

                # 根据关键词出现的频率加分
                frequency = document_text.count(keyword)
                keyword_score += min(0.05 * frequency, 0.2)  # 最大值限制为 0.2

        # 通过结合基础分数和关键词分数计算最终得分
        final_score = base_score + keyword_score

        # 将评分结果添加到列表中
        scored_results.append({
            "text": result["text"],
            "metadata": result["metadata"],
            "similarity": result["similarity"],
            "relevance_score": final_score
        })

    # 按最终相关性分数降序对结果进行排序
    reranked_results = sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)

    # 返回前 top_n 个结果
    return reranked_results[:top_n]


def generate_response(query, context):
    """
    根据查询和上下文生成响应。

    Args:
        query (str): 用户查询
        context (str): 获取到的上下文

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

    # 使用指定的模型生成响应
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回生成的响应内容
    return response.choices[0].message.content


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
