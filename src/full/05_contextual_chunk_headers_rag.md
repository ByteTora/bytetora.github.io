# 上下文块标题（CCH）

通过在生成响应之前检索相关外部知识，检索增强生成（RAG）提高了语言模型的事实准确性。然而，标准的分块往往丢失重要的上下文，使得检索效果不佳。
上下文块标题（CCH）通过在每个块嵌入之前添加高级上下文（如文档标题或部分标题）来增强 RAG。这提高了检索质量，并防止了不相关的响应。

------
实现步骤：
- 数据采集：从 PDF 中提取文本
- **带上下文标题的块分割：提取章节标题（或使用模型为块生成标题）并将其添加到块的开头。**
- 嵌入创建：将文本块转换为数值表示
- 语义搜索：根据用户查询检索相关块
- 回答生成：使用语言模型根据检索到的上下文生成回答。
- 评估：使用评估数据集评估模型性能。


```python
import fitz
import os
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
```

    python-dotenv could not parse statement starting at line 1
    python-dotenv could not parse statement starting at line 2
    python-dotenv could not parse statement starting at line 3
    python-dotenv could not parse statement starting at line 4
    python-dotenv could not parse statement starting at line 5





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

# 提取文本并识别章节标题

从 PDF 中提取文本，同时识别章节标题（块的可能标题）


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

## 文本分块与上下文标题

为了提高检索效率，我们使用LLM模型为每个片段生成描述性标题


```python
def generate_chunk_header(chunk):
    """
    使用 LLM 为给定的文本块生成标题/页眉

    Args:
        chunk (str): T要总结为标题的文本块
        model (str): 用于生成标题的模型

    Returns:
        str: 生成的标题/页眉
    """
    # 定义系统提示
    system_prompt = "为给定的文本生成一个简洁且信息丰富的标题。"

    # 根据系统提示和文本块生成
    response = client.chat.completions.create(
        model=llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
    )

    # 返回生成的标题/页眉，去除任何前导或尾随空格
    return response.choices[0].message.content.strip()
```


```python
def chunk_text_with_headers(text, n, overlap):
    """
    将文本分割为较小的片段，并生成标题。

    Args:
        text (str): 要分块的完整文本
        n (int): 每个块的字符数
        overlap (int): 块之间的重叠字符数

    Returns:
        List[dict]: 包含 'header' 和 'text' 键的字典列表
    """
    chunks = []

    # 按指定的块大小和重叠量遍历文本
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]
        header = generate_chunk_header(chunk)  # 使用 LLM 为块生成标题
        chunks.append({"header": header, "text": chunk})  # 将标题和块添加到列表中

    return chunks
```

# 从 PDF 文件中提取和分块文本


```python
extracted_text = extract_text_from_pdf(pdf_path)

# Chunk the extracted text with headers
# We use a chunk size of 1000 characters and an overlap of 200 characters
text_chunks = chunk_text_with_headers(extracted_text, 1000, 200)

# Print a sample chunk with its generated header
print("Sample Chunk:")
print("Header:", text_chunks[0]['header'])
print("Content:", text_chunks[0]['text'])
```

    Sample Chunk:
    Header: 人工智能基础：从概念到核心技术
    Content: 理解⼈⼯智能
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
    


# 为标题和文本创建嵌入


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
    # batch_size = 64
    # embeddings = []
    #
    # for i in range(0, len(texts), batch_size):
    #     batch = texts[i:i + batch_size]
    #     response = client.embeddings.create(
    #         model=embedding_model,
    #         input=batch
    #     )
    #     # 将响应转换为numpy数组列表并添加到embeddings列表中
    #     embeddings.extend([np.array(embedding.embedding) for embedding in response.data])
    #
    # return embeddings
    response = client.embeddings.create(
        model=embedding_model,
        input=texts
    )
    return response.data[0].embedding


# response = create_embeddings(text_chunks)
```


```python
embeddings = []  # Initialize an empty list to store embeddings

# Iterate through each text chunk with a progress bar
for chunk in tqdm(text_chunks, desc="Generating embeddings"):
    # Create an embedding for the chunk's text
    text_embedding = create_embeddings(chunk["text"])
    # print(text_embedding.shape)
    # Create an embedding for the chunk's header
    header_embedding = create_embeddings(chunk["header"])
    # Append the chunk's header, text, and their embeddings to the list
    embeddings.append({"header": chunk["header"], "text": chunk["text"], "embedding": text_embedding,
                       "header_embedding": header_embedding})
```

    Generating embeddings: 100%|██████████| 13/13 [00:04<00:00,  2.78it/s]


# 语义搜索


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
def semantic_search(query, chunks, k=5):
    """
    根据查询搜索最相关的块

    Args:
    query (str): 用户查询
    chunks (List[dict]): 带有嵌入的文本块列表
    k (int): 返回的相关chunk数

    Returns:
    List[dict]: Top-k most relevant chunks.
    """
    query_embedding = create_embeddings(query)
    # print(query_embedding)
    # print(query_embedding.shape)

    similarities = []

    # 遍历每个块以计算相似度分数
    for chunk in chunks:
        # Compute cosine similarity between query embedding and chunk text embedding
        sim_text = cosine_similarity(np.array(query_embedding), np.array(chunk["embedding"]))
        # sim_text = cosine_similarity(query_embedding, chunk["embedding"])

        # Compute cosine similarity between query embedding and chunk header embedding
        sim_header = cosine_similarity(np.array(query_embedding), np.array(chunk["header_embedding"]))
        # sim_header = cosine_similarity(query_embedding, chunk["header_embedding"])
        # 计算平均相似度分数
        avg_similarity = (sim_text + sim_header) / 2
        # Append the chunk and its average similarity score to the list
        similarities.append((chunk, avg_similarity))

    # Sort the chunks based on similarity scores in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    # Return the top-k most relevant chunks
    return [x[0] for x in similarities[:k]]
```

# 查询


```python
# Load validation data
with open('../../data/val.json', encoding="utf-8") as f:
    data = json.load(f)

query = data[0]['question']

# Retrieve the top 2 most relevant text chunks
top_chunks = semantic_search(query, embeddings, k=2)

# Print the results
print("Query:", query)
for i, chunk in enumerate(top_chunks):
    print(f"Header {i+1}: {chunk['header']}")
    print(f"Content:\n{chunk['text']}\n")
```

    Query: 什么是‘可解释人工智能’，为什么它被认为很重要？
    Header 1: 人工智能的未来发展与伦理挑战
    Content:
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
    ⼈⼯智能与机器⼈技术的融合，将机器⼈的物理能⼒与⼈⼯智能的认知能⼒完美结合。这种协同效
    应使机器⼈能够执⾏复杂的任务，适应不断变化的环境，并与⼈类更⾃然地互动。⼈⼯智能机器⼈
    ⼴泛应⽤于制造业、医疗保健、物流和勘探领域。
    机器⼈的类型
    ⼯业机器⼈
    ⼯业机器⼈在制造业中⽤于执⾏焊接、喷漆、装配和物料搬运等任务。⼈⼯智能提升了它们的精
    度、效率和适应性，使它们能够在协作环境中与⼈类并肩⼯作（协作机器⼈）。
    服务机器⼈
    服务机器⼈协助⼈类完成各种任务，包括清洁、送货、客⼾服务和医疗
    
    Header 2: 人工智能在各行业的应用与伦理挑战
    Content:
    改变交通运输。⾃动驾驶
    汽⻋利⽤⼈  ⼯智能感知周围环境、做出驾驶决策并安全⾏驶。
    零售
    零售⾏业利⽤⼈⼯智能进⾏个性化推荐、库存管理、客服聊天机器⼈和供应链优化。⼈⼯智能系统
    可以分析客⼾数据，预测需求、提供个性化优惠并改善购物体验。
    制造业
    ⼈⼯智能在制造业中⽤于预测性维护、质量控制、流程优化和机器⼈技术。⼈⼯智能系统可以监控
    设备、检测异常并⾃动执⾏任务，从⽽提⾼效率并降低成本。
    教育
    ⼈⼯智能正在通过个性化学习平台、⾃动评分系统和虚拟导师提升教育⽔平。⼈⼯智能⼯具可以适
    应学⽣的个性化需求，提供反馈，并打造定制化的学习体验。
    娱乐
    娱乐⾏业将⼈⼯智能⽤于内容推荐、游戏开发和虚拟现实体验。⼈⼯智能算法分析⽤⼾偏好，推荐
    电影、⾳乐和游戏，从⽽增强⽤⼾参与度。
    ⽹络安全
    ⼈⼯智能在⽹络安全领域⽤于检测和应对威胁、分析⽹络流量以及识别漏洞。⼈⼯智能系统可以⾃
    动执⾏安全任务，提⾼威胁检测的准确性，并增强整体⽹络安全态势。
    第四章：⼈⼯智能的伦理和社会影响
    ⼈⼯智能的快速发展和部署引发了重⼤的伦理和社会担忧。这些担忧包括：
    偏⻅与公平
    ⼈⼯智能系统可能会继承并放⼤其训练数据中存在的偏⻅，从⽽导致不公平或歧视性的结果。确保
    ⼈⼯智能系统的公平性并减少偏⻅是⼀项关键挑战。
    透明度和可解释性
    许多⼈⼯智能系统，尤其是深度学习模型，都是“⿊匣⼦”，很难理解它们是如何做出决策的。增
    强透明度和可解释性对于建⽴信任和问责⾄关重要。
    隐私和安全
    ⼈⼯智能系统通常依赖⼤量数据，这引发了⼈们对隐私和数据安全的担忧。保护敏感信息并确保负
    责任的数据处理⾄关重要。
    ⼯作岗位流失
    ⼈⼯智能的⾃动化能⼒引发了⼈们对⼯作岗位流失的担忧，尤其是在重复性或常规性任务的⾏业。
    应对⼈⼯智能驱动的⾃动化带来的潜在经济和社会影响是⼀项关键挑战。
    ⾃主与控制
    随着⼈⼯智能系统⽇益⾃主，控制、问责以及潜在意外后果的问题也随之⽽来。为⼈⼯智能的开发
    和部署建⽴清晰的指导⽅针和道德框架⾄关重要。
    ⼈⼯智能武器化
    ⼈⼯智能在⾃主武器系统中的潜在应⽤引发了重⼤的伦理和安全担忧。需要开展国际讨论并制定相
    关法规，以应对⼈⼯智能武器的相关⻛险。
    第五章：⼈⼯智能的未来
    ⼈⼯智能的未来很可能以持续进步和在各个领域的⼴泛应⽤为特征。关键趋势和发展领域包括：
    可解释⼈⼯智能（XAI）
    可解释⼈⼯智能 (XAI) 旨在使⼈⼯智
    


# 基于检索到的片段生成回答


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
     可解释人工智能（XAI）旨在使人工智能系统更加透明易懂。XAI 技术正在开发中，旨在深入了解人工智能模型的决策方式，从而增强信任度和责任感。它被认为很重要，因为许多人工智能系统，尤其是深度学习模型，都是“黑匣子”，很难理解它们是如何做出决策的。增强透明度和可解释性对于建立信任和问责至关重要。


# 评估


```python
evaluate_system_prompt = "你是一个智能评估系统，负责评估AI助手的回答。如果AI助手的回答与真实答案非常接近，则评分为1。如果回答错误或与真实答案不符，则评分为0。如果回答部分符合真实答案，则评分为0.5。"

evaluation_prompt = f"用户问题: {query}\nAI回答:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# Generate the evaluation response using the evaluation system prompt and evaluation prompt
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print(evaluation_response)
```

    1 
    
    AI助手的回答与真实答案非常接近。两者都强调了可解释人工智能（XAI）的目标是提高透明度和可理解性，以及其重要性在于建立信任、问责制和公平性。AI助手的回答还额外提到了深度学习模型的“黑匣子”特性，这进一步丰富了回答内容。因此，评分为1。

