"""
假设文档生成 核心函数
"""
import os
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
# 假设文档 RAG 完整流程
####################################
def hyde_rag(query, vector_store, k=5, should_generate_response=True):
    """
    使用假设文档嵌入（Hypothetical Document Embedding）执行 RAG（检索增强生成）。

    参数:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档片段的向量存储
        k (int): 要检索的片段数量
        generate_response (bool): 是否生成最终响应

    返回:
        Dict: 结果，包括假设文档和检索到的片段
    """
    print(f"\n=== 使用 HyDE 处理查询: {query} ===\n")

    # 第 1 步：生成一个假设文档来回答查询
    print("生成假设文档...")
    hypothetical_doc = generate_hypothetical_document(query)
    print(f"生成了长度为 {len(hypothetical_doc)} 个字符的假设文档")

    # 第 2 步：为假设文档创建嵌入
    print("为假设文档创建嵌入...")
    hypothetical_embedding = create_embeddings([hypothetical_doc])[0]

    # 第 3 步：根据假设文档检索相似的片段
    print(f"检索 {k} 个最相似的片段...")
    retrieved_chunks = vector_store.similarity_search(hypothetical_embedding, k=k)

    # 准备结果字典
    results = {
        "query": query,
        "hypothetical_document": hypothetical_doc,
        "retrieved_chunks": retrieved_chunks
    }

    # 第 4 步：如果需要，生成最终响应
    if should_generate_response:
        print("生成最终响应...")
        response = generate_response(query, retrieved_chunks)
        results["response"] = response

    return results


####################################
# 根据用户查询生成假设性文档
####################################
def generate_hypothetical_document(query, desired_length=1000):
    """
    生成能够回答查询的假设文档

    Args:
        query (str): 用户查询内容
        desired_length (int): 目标文档长度（字符数）

    Returns:
        str: 生成的假设文档文本
    """
    # 定义系统提示词以指导模型生成文档的方法
    system_prompt = f"""你是一位专业的文档创建专家。
    给定一个问题，请生成一份能够直接解答该问题的详细文档。
    文档长度应约为 {desired_length} 个字符，需提供深入且具有信息量的答案。
    请以权威资料的口吻撰写，内容需包含具体细节、事实和解释。
    不要提及这是假设性文档 - 直接输出内容即可。"""

    # 用查询定义用户提示词
    user_prompt = f"问题: {query}\n\n生成一份完整解答该问题的文档："

    # 调用OpenAI API生成假设文档
    response = client.chat.completions.create(
        model=llm_model,  # 指定使用的模型
        messages=[
            {"role": "system", "content": system_prompt},  # 系统指令引导模型行为
            {"role": "user", "content": user_prompt}  # 包含用户查询的输入
        ],
        temperature=0.1  # 控制输出随机性的温度参数
    )

    # 返回生成的文档内容
    return response.choices[0].message.content


def generate_response(query, relevant_chunks):
    """
    根据查询和相关文本块生成最终回答。

    Args:
        query (str): 用户查询
        relevant_chunks (List[Dict]): 检索到的相关文本块列表

    Returns:
        str: 生成的回答内容
    """
    # 将多个文本块的内容拼接起来，形成上下文
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

    # 使用 OpenAI API 生成回答
    response = client.chat.completions.create(
        model=llm_model,  # 指定使用的模型
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手。请基于提供的上下文回答问题。"},
            {"role": "user", "content": f"上下文内容：\n{context}\n\n问题：{query}"}
        ],
        temperature=0.5,  # 控制生成内容的随机性
        max_tokens=500  # 最大生成 token 数量
    )

    # 返回生成的回答内容
    return response.choices[0].message.content


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
