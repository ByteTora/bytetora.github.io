# 上下文增强检索

检索增强生成（RAG）通过从外部来源检索相关知识来增强 AI 的响应。传统的检索方法返回孤立的文本片段，这可能导致答案不完整。
为了解决这个问题，我们引入了上下文增强检索，确保检索到的信息包括相邻的片段，以实现更好的连贯性。

------
实现步骤：
- 数据采集：从 PDF 中提取文本
- 重叠上下文分块：将文本分割成重叠的块以保留上下文
- 嵌入创建：将文本块转换为数值表示
- 上下文感知检索：检索相关块及其邻居以获得更好的完整性
- 回答生成：使用语言模型根据检索到的上下文生成回答。
- 评估：使用评估数据集评估模型性能。


```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
```




    True



# 从 PDF 文件中提取文本


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

# 对提取的文本进行分块


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
        chunks.append(text[i:i + n])

    return chunks  # Return the list of text chunks

```

# 设置 OpenAI API 客户端



```python
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
```

# 从 PDF 文件中提取和分块文本

加载 PDF，提取文本，并将其分割成块。


```python
# Define the path to the PDF file
pdf_path = "../../data/AI_Information.en.zh-CN.pdf"

# Extract text from the PDF file
extracted_text = extract_text_from_pdf(pdf_path)

# Chunk the extracted text into segments of 1000 characters with an overlap of 200 characters
text_chunks = chunk_text(extracted_text, 1000, 200)

# Print the number of text chunks created
print("Number of text chunks:", len(text_chunks))

# Print the first text chunk
print("\nFirst text chunk:")
print(text_chunks[0])
```

    Number of text chunks: 13
    
    First text chunk:
    理解⼈⼯智能
    第⼀章：⼈⼯智能简介
    ⼈⼯智能 (AI) 是指数字计算机或计算机控制的机器⼈执⾏通常与智能⽣物相关的任务的能⼒。该术
    语通常⽤于开发具有⼈类特有的智⼒过程的系统，例如推理、发现意义、概括或从过往经验中学习
    的能⼒。在过去的⼏⼗年中，计算能⼒和数据可⽤性的进步显著加速了⼈⼯智能的开发和部署。
    历史背景
    ⼈⼯智能的概念已存在数个世纪，经常出现在神话和⼩说中。然⽽，⼈⼯智能研究的正式领域始于
    20世纪中叶。1956年的达特茅斯研讨会被⼴泛认为是⼈⼯智能的发源地。早期的⼈⼯智能研究侧
    重于问题解决和符号⽅法。20世纪80年代专家系统兴起，⽽20世纪90年代和21世纪初，机器学习
    和神经⽹络取得了进步。深度学习的最新突破彻底改变了这⼀领域。
    现代观察
    现代⼈⼯智能系统在⽇常⽣活中⽇益普及。从 Siri 和 Alexa 等虚拟助⼿，到流媒体服务和社交媒体
    上的推荐算法，⼈⼯智能正在影响我们的⽣活、⼯作和互动⽅式。⾃动驾驶汽⻋、先进的医疗诊断
    技术以及复杂的⾦融建模⼯具的发展，彰显了⼈⼯智能应⽤的⼴泛性和持续增⻓。此外，⼈们对其
    伦理影响、偏⻅和失业的担忧也⽇益凸显。
    第⼆章：⼈⼯智能的核⼼概念
    机器学习
    机器学习 (ML) 是⼈⼯智能的⼀个分⽀，专注于使系统⽆需明确编程即可从数据中学习。机器学习
    算法能够识别模式、做出预测，并随着接触更多数据⽽不断提升其性能。
    监督学习
    在监督学习中，算法基于标记数据进⾏训练，其中输⼊数据与正确的输出配对。这使得算法能够学
    习输⼊和输出之间的关系，并对新的、未知的数据进⾏预测。⽰例包括图像分类和垃圾邮件检测。
    ⽆监督学习
    ⽆监督学习算法基于未标记数据进⾏训练，算法必须在没有明确指导的情况下发现数据中的模式和
    结构。常⽤技术包括聚类（将相似的数据点分组）和降维（在保留重要信息的同时减少变量数
    量）。
    从英语翻译成中⽂(简体) - www.onlinedoctranslator.com
    强化学习
    强化学习涉及训练代理在特定环境中做出决策，以最⼤化奖励。代理通过反复试验进⾏学习，并以
    奖励或惩罚的形式接收反馈。这种⽅法应⽤于游戏、机器⼈技术和资源管理。
    深度学习
    深度学习是机器学习的⼀个⼦领域，它使⽤多层⼈⼯神经⽹络（深度神经⽹络）来分析数据。这些
    ⽹络的设计灵感来源于⼈脑的结构和功能。深度学习在图像识别、⾃然语⾔处理和语⾳识别等领域
    


# 创建嵌入


```python
def create_embeddings(texts):
    """
    为文本列表生成嵌入

    Args:
    texts (List[str]): 输入文本列表.

    Returns:
    List[np.ndarray]: List of numerical embeddings.
    """
    # 确保每次调用不超过64条文本
    batch_size = 64
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL_ID"),
            input=batch
        )
        # 将响应转换为numpy数组列表并添加到embeddings列表中
        embeddings.extend([np.array(embedding.embedding) for embedding in response.data])

    return embeddings

response = create_embeddings(text_chunks)
```

# 实现上下文感知语义搜索

修改检索以包含相邻块以获得更好的上下文


```python
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): First vector.
    vec2 (np.ndarray): Second vector.

    Returns:
    float: Cosine similarity score.
    """

    # Compute the dot product of the two vectors
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```


```python
def context_enriched_search(query, text_chunks, embeddings, k=1, context_size=1):
    """
    检索最相关的文本块及其相邻的上下文块

    Args:
    query (str): 搜索查询
    text_chunks (List[str]): 文本块列表
    embeddings (List[dict]): 文本块嵌入列表
    k (int): 要检索的相关块数量
    context_size (int): 包含的相邻块数量

    Returns:
    List[str]: 包含上下文信息的相关文本块
    """
    # 将查询转换为嵌入向量
    query_embedding = create_embeddings(query)
    similarity_scores = []

    # 计算查询与每个文本块嵌入之间的相似度分数
    for i, chunk_embedding in enumerate(embeddings):
        # 计算查询嵌入与当前文本块嵌入之间的余弦相似度
        similarity_score = cosine_similarity(np.array(query_embedding), chunk_embedding)
        # 将索引和相似度分数存储为元组
        similarity_scores.append((i, similarity_score))

    # 按相似度分数降序排序（相似度最高排在前面）
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 获取最相关块的索引
    # top_index = [index for index, _ in similarity_scores[:k]]
    top_index = similarity_scores[0][0]

    # similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    # top_indices = np.argsort(similarities)[-k:][::-1]

    # 定义上下文包含的范围
    # 确保不会超出 text_chunks 的边界
    start = max(0, top_index - context_size)
    end = min(len(text_chunks), top_index + context_size + 1)

    # 返回最相关的块及其相邻的上下文块
    return [text_chunks[i] for i in range(start, end)]
```

# 带有上下文检索的查询


```python
# 从 JSON 文件加载验证数据集
with open('../../data/val.json', encoding="utf-8") as f:
    data = json.load(f)

# 从数据集中提取第一个问题作为查询
query = data[0]['question']

# 检索最相关的块及其相邻的上下文块
# Parameters:
# - query: 我们要搜索的问题
# - text_chunks: 从 PDF 中提取的文本块
# - response.data: 文本块的嵌入
# - k=1: 返回最佳匹配
# - context_size=1: 包括最佳匹配前后的各一个块作为上下文
top_chunks = context_enriched_search(query, text_chunks, response, k=1, context_size=1)

# Print the query for reference
print("Query:", query)
# 打印每个检索到的块，并附上标题和分隔符
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")
```

    Query: 什么是‘可解释人工智能’，为什么它被认为很重要？
    Context 1:
    系统可以分析数据、预测⽓候影响并为缓解策略提供信息。
    精准农业
    ⼈⼯智能通过监测作物、优化资源利⽤和预测产量来增强精准农业。⼈⼯智能⼯具可以改善耕作⽅
    式，减少环境影响，并增强粮⻝安全。
    野⽣动物保护
    ⼈⼯智能在野⽣动物保护领域发挥着重要作⽤，它可以监测野⽣动物种群数量、追踪其活动轨迹，
    并侦测偷猎活动。⼈⼯智能系统可以分析来⾃传感器、摄像头和⽆⼈机的数据，为保护⼯作提供宝
    贵的洞⻅。
    环境监测
    ⼈⼯智能环境监测系统可以追踪空⽓和⽔质，检测污染，并⽀持环境保护⼯作。这些系统提供实时
    数据，识别污染源，并为环境政策提供信息。
    灾难响应
    ⼈⼯智能通过分析数据、预测影响和⽀持救援⼯作来增强灾难响应能⼒。⼈⼯智能系统可以提⾼态
    势感知能⼒，优化资源配置，并增强救援⼈员之间的协调。
    第 18 章：政府和政策在⼈⼯智能中的作⽤
    ⼈⼯智能战略与政策框架
    世界各国政府正在制定⼈⼯智能战略和政策框架，以指导⼈⼯智能的开发和部署。这些框架旨在解
    决伦理问题，促进创新，并确保负责任的⼈⼯智能实践。
    ⼈⼯智能监管
    ⼈⼯智能监管是⼀个复杂且不断发展的领域。各国政府正在考虑制定法规，以解决偏⻅、透明度、
    隐私和安全等问题。平衡创新与伦理考量是⼀项关键挑战。
    ⼈⼯智能研发资⾦
    政府在资助⼈⼯智能研发⽅⾯发挥着⾄关重要的作⽤。公共资⾦⽀持基础研究、应⽤研究以及⼈⼯
    智能基础设施的建设。政府投资推动创新，促进合作。
    国际合作
    国际合作对于应对⼈⼯智能带来的全球挑战和机遇⾄关重要。这包括共享知识、制定标准以及跨境
    推⼴负责任的⼈⼯智能实践。
    公众参与和教育
    让公众参与⼈⼯智能讨论，对于建⽴信任并确保⼈⼯智能发展符合社会价值观⾄关重要。教育和宣
    传活动可以让公众了解⼈⼯智能、其影响及其潜⼒。
    第 19 章：⼈⼯智能与伦理
    道德⼈⼯智能原则
    符合伦理道德的⼈⼯智能原则指导着⼈⼯智能系统的开发和部署，以确保其公平、透明、负责且有
    益于社会。关键原则包括尊重⼈权、隐私、不歧视和仁慈。
    解决⼈⼯智能中的偏⻅
    ⼈⼯智能系统可能会继承并放⼤其训练数据中存在的偏⻅，从⽽导致不公平或歧视性的结果。解决
    偏⻅需要谨慎的数据收集、算法设计以及持续的监测和评估。
    透明度和可解释性
    透明度和可解释性对于建⽴对⼈⼯智能系统的信任⾄关重要。可解释⼈⼯智能 (XAI) 技术旨在使⼈
    ⼯智能决策更易于理解，使⽤⼾能够评估其公平性和准确
    =====================================
    Context 2:
    透明、负责且有
    益于社会。关键原则包括尊重⼈权、隐私、不歧视和仁慈。
    解决⼈⼯智能中的偏⻅
    ⼈⼯智能系统可能会继承并放⼤其训练数据中存在的偏⻅，从⽽导致不公平或歧视性的结果。解决
    偏⻅需要谨慎的数据收集、算法设计以及持续的监测和评估。
    透明度和可解释性
    透明度和可解释性对于建⽴对⼈⼯智能系统的信任⾄关重要。可解释⼈⼯智能 (XAI) 技术旨在使⼈
    ⼯智能决策更易于理解，使⽤⼾能够评估其公平性和准确性。
    隐私和数据保护
    ⼈⼯智能系统通常依赖⼤量数据，这引发了⼈们对隐私和数据保护的担忧。确保负责任的数据处
    理、实施隐私保护技术以及遵守数据保护法规⾄关重要。
    问责与责任
    建⽴⼈⼯智能系统的问责制和责任制，对于应对潜在危害和确保道德⾏为⾄关重要。这包括明确⼈
    ⼯智能系统开发者、部署者和⽤⼾的⻆⾊和职责。
    第 20 章：建⽴对⼈⼯智能的信任
    透明度和可解释性
    透明度和可解释性是建⽴⼈⼯智能信任的关键。让⼈⼯智能系统易于理解，并深⼊了解其决策过
    程，有助于⽤⼾评估其可靠性和公平性。
    稳健性和可靠性
    确保⼈⼯智能系统的稳健可靠对于建⽴信任⾄关重要。这包括测试和验证⼈⼯智能模型、监控其性
    能以及解决潜在的漏洞。
    ⽤⼾控制和代理
    赋予⽤⼾对AI系统的控制权，并赋予他们与AI交互的⾃主权，可以增强信任。这包括允许⽤⼾⾃定
    义AI设置、了解其数据的使⽤⽅式，以及选择退出AI驱动的功能。
    道德设计与发展
    将伦理考量纳⼊⼈⼯智能系统的设计和开发对于建⽴信任⾄关重要。这包括进⾏伦理影响评估、与
    利益相关者沟通，以及遵守伦理准则和标准。
    公众参与和教育
    让公众参与⼈⼯智能的讨论，并教育他们了解其能⼒、局限性和伦理影响，有助于建⽴信任。公众
    意识宣传活动、教育计划和开放式对话有助于促进公众对⼈⼯智能的理解和接受。
    第 21 章：⼈⼯智能的前进之路
    持续研究与创新
    持续的研究和创新对于提升⼈⼯智能能⼒、应对挑战并充分发挥其潜⼒⾄关重要。这包括投资基础
    研究、应⽤研究以及新型⼈⼯智能技术和应⽤的开发。
    负责任的开发和部署
    负责任地开发和部署⼈⼯智能对于确保其效益得到⼴泛共享并降低其⻛险⾄关重要。这涉及遵守伦
    理原则、促进公平透明以及保护⼈权和价值观。
    全球协作与合作
    全球协作与合作对于应对⼈⼯智能带来的全球挑战和机遇⾄关重要。这包括共享知识、制定标准以
    及跨境推⼴负责任的⼈⼯智能实践。
    教育和劳动⼒发
    =====================================
    Context 3:
    挑战并充分发挥其潜⼒⾄关重要。这包括投资基础
    研究、应⽤研究以及新型⼈⼯智能技术和应⽤的开发。
    负责任的开发和部署
    负责任地开发和部署⼈⼯智能对于确保其效益得到⼴泛共享并降低其⻛险⾄关重要。这涉及遵守伦
    理原则、促进公平透明以及保护⼈权和价值观。
    全球协作与合作
    全球协作与合作对于应对⼈⼯智能带来的全球挑战和机遇⾄关重要。这包括共享知识、制定标准以
    及跨境推⼴负责任的⼈⼯智能实践。
    教育和劳动⼒发展
    为迎接⼈⼯智能的未来，需要开展教育和培训计划，使个⼈掌握操作⼈⼯智能系统和适应新⼯作⻆
    ⾊所需的技能。这包括推⼴ STEM 教育、提供再培训和技能提升机会，以及培养终⾝学习的习惯。
    以⼈为本的⽅法
    以⼈为本的⼈⼯智能⽅法侧重于开发能够增强⼈类能⼒、促进福祉并符合⼈类价值观的⼈⼯智能系
    统。这需要考量⼈⼯智能的伦理、社会和⼼理影响，并优先考虑⼈类的需求和利益。
    通过秉持这些原则并携⼿合作，我们能够充分利⽤⼈⼯智能的变⾰潜⼒，创造⼀个更具创新性、公
    平性和可持续性的未来。前进的道路需要奉献、协作以及对负责任的⼈⼯智能开发和部署的承诺。
    
    =====================================


# 基于检索到的片段生成响应



```python
# AI 助手的系统提示
system_prompt = "你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"

def generate_response(system_prompt, user_prompt):
    """
    基于检索到的文本块生成 AI 回答。

    Args:
    retrieved_chunks (List[str]): 检索到的文本块列表
    model (str): AI model.

    Returns:
    str: AI-generated response.
    """
    # Generate the AI response using the specified model
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_ID"),
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Return the content of the AI response
    return response.choices[0].message.content

# 将检索到的文本块合并为一个上下文字符串
context = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])

# 通过组合上下文和查询创建用户提示
user_prompt = f"{context}\n\nQuestion: {query}"
ai_response = generate_response(system_prompt, user_prompt)
print("AI Response:\n", ai_response)
```

    AI Response:
     **可解释人工智能（XAI）**是指那些能够使其决策过程和结果易于理解和解释的人工智能系统。它旨在让用户、开发者和其他利益相关者能够清楚地了解AI系统是如何做出特定决策或预测的。
    
    **为什么可解释人工智能很重要？**
    
    1. **建立信任**：透明度和可解释性有助于用户理解和信任AI系统，知道其决策是基于合理的逻辑和可靠的数据。
    
    2. **评估公平性和准确性**：用户能够评估AI系统的决策是否公平、无偏见，并且准确无误。
    
    3. **伦理和法律责任**：在涉及伦理和法律问题时，能够解释AI的决策过程至关重要，以确保其符合伦理标准和法律法规。
    
    4. **问题诊断和改进**：当AI系统出现错误或偏差时，可解释性有助于开发者快速诊断问题并进行改进。
    
    5. **用户控制和代理**：赋予用户对AI系统的控制权，让他们能够根据解释的信息做出更明智的决策。
    
    总的来说，可解释人工智能是确保AI系统透明、负责且有益于社会的关键因素。


# 评估响应质量


```python
evaluate_system_prompt = "你是一个智能评估系统，负责评估AI助手的回答。如果AI助手的回答与真实答案非常接近，则评分为1。如果回答错误或与真实答案不符，则评分为0。如果回答部分符合真实答案，则评分为0.5。"

evaluation_prompt = f"用户问题: {query}\nAI回答:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# Generate the evaluation response using the evaluation system prompt and evaluation prompt
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print(evaluation_response)
```

    **可解释人工智能（XAI）**是指那些能够使其决策过程和结果易于理解和解释的人工智能系统。与传统的人工智能系统相比，XAI不仅提供答案或结果，还能清晰地展示其推理过程、所依赖的数据和算法逻辑。
    
    **为什么可解释人工智能很重要？**
    
    1. **建立信任**：透明度和可解释性有助于用户理解和信任AI系统的决策，特别是在高风险领域（如医疗、金融和法律）。
    
    2. **检测和解决偏见**：通过解释决策过程，可以更容易地识别和纠正数据或算法中的偏见，从而避免不公平或歧视性的结果。
    
    3. **提高决策质量**：用户可以更好地评估AI建议的合理性和准确性，从而做出更明智的决策。
    
    4. **符合法规要求**：许多国家和地区正在制定法规，要求AI系统具备一定的透明度和可解释性，以保护用户权益。
    
    5. **促进伦理和责任**：可解释性有助于明确AI系统开发者、部署者和用户的角色和责任，确保道德行为。
    
    综上所述，可解释人工智能在增强用户信任、确保公平性、提高决策质量、符合法规和促进伦理责任方面具有重要意义。
    
    **评分**：1
    
    AI助手的回答准确且全面地解释了“可解释人工智能”的概念及其重要性，与真实答案非常接近。

