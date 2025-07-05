"""
纠错型检索 核心函数
"""
import os
import re
import requests
from urllib.parse import quote_plus
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
import requests

# proxies = {
#     "https": "127.0.0.1:9090",
#     "http": "127.0.0.1:9090"
# }


client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
llm_model = os.getenv("LLM_MODEL_ID")
embedding_model = os.getenv("EMBEDDING_MODEL_ID")


####################################
# CRAG（纠正型检索RAG） 完整流程
####################################
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


####################################
# 评估文档与查询的相关性：给出一个评估分数
####################################
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


####################################
# 从文本中提取并精炼关键信息：使检索到的信息减少冗余
####################################
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


####################################
# 查询改写查询：对查询进行重写，以提高搜索效果
####################################
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


####################################
# 查询改写
####################################
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


####################################
# 使用 DuckDuckGo 进行网络搜索
####################################
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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        proxies = {
            "http": "socks5://127.0.0.1:9090",
            "https": "socks5://127.0.0.1:9090"
        }
        response = requests.get(url,
                                headers=headers,
                                proxies=proxies
                                )
        print(response)

        data = response.json()
        # print(data)

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


if __name__ == '__main__':
    from duckduckgo_search import DDGS
    query = "机器学习 vs 传统编程 区别"
    # ddgs = DDGS(proxy="127.0.0.1:9090", timeout=20)  # "tb" is an alias for "socks5://127.0.0.1:9150"
    # results = ddgs.text(query, max_results=5)
    # print(results)
    res_text, source = duck_duck_go_search(query)
    print(res_text)