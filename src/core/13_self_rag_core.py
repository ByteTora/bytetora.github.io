"""
Self-RAG 核心函数
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
# Self-RAG 完整流程
####################################
def self_rag(query, vector_store, top_k=3):
    """
    实现完整的Self-RAG流程。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        top_k (int): 初始检索的文档数量

    Returns:
        dict: 包括查询、响应和Self-RAG流程指标的结果
    """
    print(f"\n=== 开始Self-RAG处理查询: {query} ===\n")

    # 第1步：确定是否需要检索
    print("第1步：确定是否需要检索...")
    retrieval_needed = determine_if_retrieval_needed(query)  # 调用函数判断是否需要检索
    print(f"是否需要检索: {retrieval_needed}")

    # 初始化指标以跟踪Self-RAG流程
    metrics = {
        "retrieval_needed": retrieval_needed,  # 是否需要检索
        "documents_retrieved": 0,  # 检索到的文档数量
        "relevant_documents": 0,  # 相关文档数量
        "response_support_ratings": [],  # 回答支持的评级列表
        "utility_ratings": []  # 回答实用性评级列表
    }

    best_response = None  # 最佳响应初始化为None
    best_score = -1  # 最佳分数初始化为-1

    if retrieval_needed:
        # 第2步：检索文档
        print("\n第2步：检索相关文档...")
        query_embedding = create_embeddings(query)  # 创建查询的嵌入向量
        results = vector_store.similarity_search(query_embedding, k=top_k)  # 搜索相似文档
        metrics["documents_retrieved"] = len(results)  # 更新检索到的文档数量
        print(f"检索到 {len(results)} 个文档")

        # 第3步：评估每个文档的相关性
        print("\n第3步：评估文档相关性...")
        relevant_contexts = []  # 初始化相关上下文列表

        for i, result in enumerate(results):
            context = result["text"]  # 提取文档内容
            relevance = evaluate_relevance(query, context)  # 评估文档与查询的相关性
            print(f"文档 {i + 1} 相关性: {relevance}")

            if relevance == "relevant":  # 如果文档相关，则添加到相关上下文列表
                relevant_contexts.append(context)

        metrics["relevant_documents"] = len(relevant_contexts)  # 更新相关文档数量
        print(f"找到 {len(relevant_contexts)} 个相关文档")

        if relevant_contexts:
            # 第4步：处理每个相关上下文
            print("\n第4步：处理相关上下文...")
            for i, context in enumerate(relevant_contexts):
                print(f"\n处理上下文 {i + 1}/{len(relevant_contexts)}...")

                # 根据上下文生成响应
                print("生成响应...")
                response = generate_response(query, context)  # 根据上下文生成响应

                # 评估响应对上下文的支持程度
                print("评估支持程度...")
                support_rating = assess_support(response, context)  # 评估支持程度
                print(f"支持评级: {support_rating}")
                metrics["response_support_ratings"].append(support_rating)  # 添加支持评级

                # 评估响应的实用性
                print("评估实用性...")
                utility_rating = rate_utility(query, response)  # 评估实用性
                print(f"实用性评级: {utility_rating}/5")
                metrics["utility_ratings"].append(utility_rating)  # 添加实用性评级

                # 计算总体评分（支持和实用性越高，评分越高）
                support_score = {
                    "fully supported": 3,  # 完全支持得分为3
                    "partially supported": 1,  # 部分支持得分为1
                    "no support": 0  # 无支持得分为0
                }.get(support_rating, 0)

                overall_score = support_score * 5 + utility_rating  # 计算总体评分
                print(f"总体评分: {overall_score}")

                # 跟踪最佳响应
                if overall_score > best_score:  # 如果当前评分高于最佳评分，则更新最佳响应和评分
                    best_response = response
                    best_score = overall_score
                    print("找到新的最佳响应！")

        # 如果没有找到相关上下文或所有响应评分较低
        if not relevant_contexts or best_score <= 0:
            print("\n未找到合适的上下文或响应评分较差，直接生成响应而不进行检索...")
            best_response = generate_response(query)  # 不使用检索直接生成响应
    else:
        # 不需要检索，直接生成响应
        print("\n不需要检索，直接生成响应...")
        best_response = generate_response(query)  # 不使用检索直接生成响应

    # 最终指标
    metrics["best_score"] = best_score  # 更新最佳评分
    metrics["used_retrieval"] = retrieval_needed and best_score > 0  # 更新是否使用了检索

    print("\n=== Self-RAG完成 ===")

    return {
        "query": query,  # 查询
        "response": best_response,  # 最佳响应
        "metrics": metrics  # 指标
    }


####################################
# 检索决策：判断给定查询是否需要检索，事实性查询检索，观点类查询不检索
####################################
def determine_if_retrieval_needed(query):
    """
    判断给定查询是否需要检索。

    Args:
        query (str): 用户查询

    Returns:
        bool: 如果需要检索，返回True；否则返回False
    """
    # 系统提示，指导AI如何判断是否需要检索
    system_prompt = """你是一个判断查询是否需要检索的AI助手。
    针对事实性问题、具体信息请求或关于事件、人物、概念的查询，回答"Yes"。
    对于观点类、假设性场景或常识性简单查询，回答"No"。
    仅回答"Yes"或"No"。"""

    # 包含查询的用户提示
    user_prompt = f"查询: {query}\n\n准确回答此查询是否需要检索？"

    # 使用模型生成响应
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从模型响应中提取答案并转换为小写
    answer = response.choices[0].message.content.strip().lower()

    # 如果答案包含“yes”，返回True；否则返回False
    return "yes" in answer


####################################
# 相关性评估：评估文本块与查询的相关性
####################################
def evaluate_relevance(query, context):
    """
    评估上下文与查询的相关性。

    Args:
        query (str): 用户查询
        context (str): 上下文文本

    Returns:
        str: 'relevant'（相关）或 'irrelevant'（不相关）
    """
    # 系统提示，指导AI如何判断文档是否与查询相关
    system_prompt = """你是一个AI助手，任务是判断文档是否与查询相关。
    判断文档中是否包含有助于回答查询的信息。
    仅回答“Relevant”或“Irrelevant”。"""

    # 如果上下文过长以避免超出标记限制，则截断上下文
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    # 包含查询和文档内容的用户提示
    user_prompt = f"""查询: {query}
    文档内容:
    {context}

    该文档与查询相关？仅回答“Relevant”或“Irrelevant”。
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

    # 从模型响应中提取答案并转换为小写
    answer = response.choices[0].message.content.strip().lower()

    return answer  # 返回相关性评估结果


####################################
# 支持性评估：评估响应是否基于给定的上下文
####################################
def assess_support(response, context):
    """
    评估响应在多大程度上得到上下文的支持。

    Args:
        response (str): 生成的响应
        context (str): 上下文文本

    Returns:
        str: 'fully supported'（完全支持）、'partially supported'（部分支持）或 'no support'（无支持）
    """
    # 系统提示，指导AI如何评估支持情况
    system_prompt = """你是一个AI助手，任务是判断回答是否基于给定的上下文。
    评估响应中的事实、主张和信息是否由上下文支持。
    仅回答以下三个选项之一：
    - "Fully supported"（完全支持）：回答所有信息均可从上下文直接得出。
    - "Partially supported"（部分支持）：回答中的部分信息由上下文支持，但部分不是。
    - "No support"（无支持）：回答中包含大量未在上下文中找到、提及或与上下文矛盾的信息。
    """

    # 如果上下文过长以避免超出标记限制，则截断上下文
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"

    # 包含上下文和要评估的响应的用户提示
    user_prompt = f"""上下文:
    {context}

    回答:
    {response}

    该回答与上下文的支持程度如何？仅回答 "Fully supported"、"Partially supported"或 "No support"。
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

    # 从模型响应中提取答案并转换为小写
    answer = response.choices[0].message.content.strip().lower()

    return answer  # 返回支持评估结果


####################################
# 效用评估：评估生成的回答是否对查询的有用，打出分数
####################################
def rate_utility(query, response):
    """
    评估响应对查询的实用性。

    Args:
        query (str): 用户查询
        response (str): 生成的响应

    Returns:
        int: 实用性评分，范围为1到5
    """
    # 系统提示，指导AI如何评估响应的实用性
    system_prompt = """你是一个AI助手，任务是评估一个回答对查询的实用性。
    从回答准确性、完整性、正确性和帮助性进行综合评分。
    使用1-5级评分标准：
    - 1：毫无用处
    - 2：稍微有用
    - 3：中等有用
    - 4：非常有用
    - 5：极其有用
    仅回答一个从1到5的单个数字，不要过多解释。"""

    # 包含查询和要评分的响应的用户提示
    user_prompt = f"""查询: {query}
    回答:
    {response}

    请用1到5分的评分评估该回答的效用，仅用一个1-5的数字评分。"""

    # 使用OpenAI客户端生成实用性评分
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从模型响应中提取评分
    rating = response.choices[0].message.content.strip()

    # 提取评分中的数字
    rating_match = re.search(r'[1-5]', rating)
    if rating_match:
        return int(rating_match.group())  # 返回提取的评分作为整数

    return 3  # 如果解析失败，默认返回中间评分


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


def generate_response(query, context=None):
    """
    根据查询和可选的上下文生成响应。

    Args:
        query (str): 用户查询
        context (str, 可选): 上下文文本

    Returns:
        str: 生成的响应
    """
    # 系统提示，指导AI如何生成有用的响应
    system_prompt = """你是一个有帮助的AI助手。请针对查询提供清晰、准确且信息丰富的回答。"""

    # 根据是否提供了上下文创建用户提示
    if context:
        user_prompt = f"""上下文:
        {context}

        查询: {query}

        请基于提供的上下文回答该查询。
        """
    else:
        user_prompt = f"""查询: {query}

        请尽你所能回答该查询。"""

    # 使用OpenAI客户端生成响应
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    # 返回生成的响应文本
    return response.choices[0].message.content.strip()
