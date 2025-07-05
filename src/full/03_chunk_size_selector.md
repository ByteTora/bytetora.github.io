# 评估simple rag 中的块大小

选择合适的块大小对于提高检索增强生成（RAG）管道的检索准确性至关重要。目标是平衡检索性能与响应质量。

-----
以下方式评估不同的块大小:
- 从 PDF 中提取文本
- 将文本分割成不同大小的块
- 为每个块创建嵌入
- 为查询检索相关块
- 使用检索到的块生成响应
- 评估响应质量
- 比较不同块大小的结果

-----
实现步骤:
- 从 PDF 中提取文本：按页获取页面文本
- 将文本分割成不同大小的块，为每个块创建嵌入
- 根据查询检索相关块
- 使用检索到的文本块用模型生成回答
- 评估不同大小块的检索回答质量

# 设置环境


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



# 设置 OpenAI API 客户端


```python
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
```

# 从 PDF 中提取文本


```python
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text

    # Iterate through each page in the PDF
    for page in mypdf:
        # Extract text from the current page and add spacing
        all_text += page.get_text("text") + " "

    # Return the extracted text, stripped of leading/trailing whitespace
    return all_text.strip()

# Define the path to the PDF file
pdf_path = "../../data/AI_Information.en.zh-CN.pdf"

# Extract text from the PDF file
extracted_text = extract_text_from_pdf(pdf_path)

# Print the first 500 characters of the extracted text
print(extracted_text[:500])
```

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
    第⼆章：⼈⼯智能


# 对提取的文本进行分块

为了提高检索效率，我们将提取的文本分割成不同大小的重叠块


```python
def chunk_text(text, n, overlap):
    """
    将文本分割为重叠的块。

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

# 定义要评估的不同块大小
chunk_sizes = [128, 256, 512]

# 创建一个字典，用于存储每个块大小对应的文本块
text_chunks_dict = {size: chunk_text(extracted_text, size, size // 5) for size in chunk_sizes}

# 打印每个块大小生成的块数量
for size, chunks in text_chunks_dict.items():
    print(f"Chunk Size: {size}, Number of Chunks: {len(chunks)}")
```

    Chunk Size: 128, Number of Chunks: 98
    Chunk Size: 256, Number of Chunks: 50
    Chunk Size: 512, Number of Chunks: 25


# 为文本片段创建嵌入

嵌入将文本转换为数值表示，以进行相似性搜索。


```python
from tqdm import tqdm
import numpy as np
import os
# 假设client已经被正确初始化和配置

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

# 假设text_chunks_dict是一个字典，键是块大小，值是文本块列表
chunk_embeddings_dict = {}
for size, chunks in tqdm(text_chunks_dict.items(), desc="Generating Embeddings"):
    chunk_embeddings_dict[size] = create_embeddings(chunks)

```

    Generating Embeddings: 100%|██████████| 3/3 [00:03<00:00,  1.15s/it]


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
def retrieve_relevant_chunks(query, text_chunks, chunk_embeddings, k=5):
    """
    检索与查询最相关的前k个文本块

    Args:
    query (str): 用户查询
    text_chunks (List[str]): 文本块列表
    chunk_embeddings (List[np.ndarray]): 文本块的嵌入列表
    k (int): 返回的前k个块数量

    Returns:
    List[str]: 最相关的文本块列表
    """
    # 为查询生成一个嵌入 - 将查询作为列表传递并获取第一个项目
    query_embedding = create_embeddings([query])[0]

    # 计算查询嵌入与每个块嵌入之间的余弦相似度
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]

    # 获取前k个最相似块的索引
    top_indices = np.argsort(similarities)[-k:][::-1]

    # 返回前k个最相关的文本块
    return [text_chunks[i] for i in top_indices]
```


```python
# 从 JSON 文件加载验证数据
with open('../../data/val.json', encoding="utf-8") as f:
    data = json.load(f)

# 从验证数据中提取第一个查询
query = data[3]['question']

# 对于每个块大小，检索相关的文本块
retrieved_chunks_dict = {size: retrieve_relevant_chunks(query, text_chunks_dict[size], chunk_embeddings_dict[size]) for size in chunk_sizes}

# 打印块大小为 256 的检索到的文本块
print(retrieved_chunks_dict[256])
```

    ['健\n医疗诊断与治疗\n⼈⼯智能正在通过分析医学影像、预测患者预后并协助制定治疗计划，彻底改变医学诊断和治疗。\n⼈⼯智能⼯具能够提⾼准确性、效率和患者护理⽔平。\n药物研发\n⼈⼯智能通过分析⽣物数据、预测药物疗效和识别潜在候选药物，加速药物的发现和开发。⼈⼯智\n能系统缩短了新疗法上市的时间并降低了成本。\n个性化医疗\n⼈⼯智能通过分析个体患者数据、预测治疗反应并制定⼲预措施，实现个性化医疗。个性化医疗可\n提⾼治疗效果并减少不良反应。\n机器⼈⼿术\n⼈⼯智能机器⼈⼿术系统能够帮助外科医⽣以更⾼的精度和控制⼒执⾏复杂的⼿', '问、提供指导并跟踪学习进度。这些\n⼯具扩⼤了教育覆盖⾯，并提升了学习成果。\n⾃动评分和反馈\n⼈⼯智能⾃动化评分和反馈流程，节省教育⼯作者的时间，并及时为学⽣提供反馈。⼈⼯智能系统\n可以评估论⽂、作业和考试，找出需要改进的地⽅。\n教育数据挖掘\n教育数据挖掘利⽤⼈⼯智能分析学⽣数据，识别学习模式并预测学习成果。这些信息可以为教学策\n略提供参考，改进教育项⽬，并增强学⽣⽀持服务。\n 第 11 章：⼈⼯智能与医疗保健\n医疗诊断与治疗\n⼈⼯智能正在通过分析医学影像、预测患者预后并协助制定治疗计划，彻底改变医学诊断和治', '果并减少不良反应。\n机器⼈⼿术\n⼈⼯智能机器⼈⼿术系统能够帮助外科医⽣以更⾼的精度和控制⼒执⾏复杂的⼿术。这些系统能够\n提⾼⼿术灵活性，减少创伤，并改善患者的治疗效果。\n医疗保健管理\n⼈⼯智能通过⾃动化任务、管理患者记录和优化⼯作流程来简化医疗保健管理。⼈⼯智能系统可以\n提⾼效率、降低成本并增强患者体验。\n第 12 章：⼈⼯智能与⽹络安全\n威胁检测与预防\n⼈⼯智能通过检测和预防威胁、分析⽹络流量以及识别漏洞来增强⽹络安全。⼈⼯智能系统可以⾃\n动执⾏安全任务，提⾼威胁检测的准确性，并增强整体⽹络安全态势。\n异', '、个性化医疗和机器⼈⼿术等应⽤改变医疗保健。⼈⼯智能\n⼯具可以分析医学图像、预测患者预后并协助制定治疗计划。\n⾦融\n在⾦融领域，⼈⼯智能⽤于欺诈检测、算法交易、⻛险管理和客⼾服务。⼈⼯智能算法可以分析⼤\n型数据集，以识别模式、预测市场趋势并实现财务流程⾃动化。\n 运输\n随着⾃动驾驶汽⻋、交通优化系统和物流管理的发展，⼈⼯智能正在彻底改变交通运输。⾃动驾驶\n汽⻋利⽤⼈  ⼯智能感知周围环境、做出驾驶决策并安全⾏驶。\n零售\n零售⾏业利⽤⼈⼯智能进⾏个性化推荐、库存管理、客服聊天机器⼈和供应链优化。⼈⼯智能系统\n', '学⽣的个性化需求，提供反馈，并打造定制化的学习体验。\n娱乐\n娱乐⾏业将⼈⼯智能⽤于内容推荐、游戏开发和虚拟现实体验。⼈⼯智能算法分析⽤⼾偏好，推荐\n电影、⾳乐和游戏，从⽽增强⽤⼾参与度。\n⽹络安全\n⼈⼯智能在⽹络安全领域⽤于检测和应对威胁、分析⽹络流量以及识别漏洞。⼈⼯智能系统可以⾃\n动执⾏安全任务，提⾼威胁检测的准确性，并增强整体⽹络安全态势。\n第四章：⼈⼯智能的伦理和社会影响\n⼈⼯智能的快速发展和部署引发了重⼤的伦理和社会担忧。这些担忧包括：\n偏⻅与公平\n⼈⼯智能系统可能会继承并放⼤其训练数据中存在的偏']


# 基于检索到的片段生成响应

基于检索到的文本为块大小 256 生成一个响应


```python
# AI 助手的系统提示
system_prompt = "你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"

def generate_response(query, system_prompt, retrieved_chunks):
    """
    基于检索到的文本块生成 AI 回答。

    Args:
    query (str): 用户查询
    retrieved_chunks (List[str]): 检索到的文本块列表
    model (str): AI model.

    Returns:
    str: AI-generated response.
    """
    # 将检索到的文本块合并为一个上下文字符串
    context = "\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

    # 通过组合上下文和查询创建用户提示
    user_prompt = f"{context}\n\nQuestion: {query}"

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

# 为每个块大小生成 AI 回答
ai_responses_dict = {size: generate_response(query, system_prompt, retrieved_chunks_dict[size]) for size in chunk_sizes}

# 打印块大小为 256 的回答
print(ai_responses_dict[256])
```

    人工智能通过以下方式为个性化医疗做出贡献：
    
    1. **分析个体患者数据**：人工智能可以处理和分析大量个体患者的健康数据，包括基因组信息、病史、生活方式等，从而深入了解每位患者的独特情况。
    
    2. **预测治疗反应**：基于对个体数据的分析，人工智能可以预测患者对不同治疗方案的响应情况，帮助医生选择最有效的治疗方法。
    
    3. **制定干预措施**：人工智能能够根据患者的具体情况，制定个性化的治疗计划和干预措施，以提高治疗效果并减少不良反应。
    
    这些功能共同提升了个性化医疗的水平，使治疗更加精准和高效。


# 评估响应质量


根据忠实度和相关性对回复进行评分


```python
# 定义评估评分系统的常量
SCORE_FULL = 1.0     # 完全匹配或完全令人满意
SCORE_PARTIAL = 0.5  # 部分匹配或部分令人满意
SCORE_NONE = 0.0     # 无匹配或不令人满意
```


```python
# 定义严格的评估提示模板
FAITHFULNESS_PROMPT_TEMPLATE = """
评估 AI 回答与真实答案的一致性、忠实度。
用户查询: {question}
AI 回答: {response}
真实答案: {true_answer}

一致性衡量 AI 回答与真实答案中的事实对齐的程度，且不包含幻觉信息。
忠实度衡量的是AI的回答在没有幻觉的情况下与真实答案中的事实保持一致的程度。

指示：
- 严格使用以下值进行评分：
    * {full} = 完全一致，与真实答案无矛盾
    * {partial} = 部分一致，存在轻微矛盾
    * {none} = 不一致，存在重大矛盾或幻觉信息
- 仅返回数值评分（{full}, {partial}, 或 {none}），无需解释或其他附加文本。
"""

```


```python
RELEVANCY_PROMPT_TEMPLATE = """
评估 AI 回答与用户查询的相关性。
用户查询: {question}
AI 回答: {response}

相关性衡量回答在多大程度上解决了用户的问题。

指示：
- 严格使用以下值进行评分：
    * {full} = 完全相关，直接解决查询
    * {partial} = 部分相关，解决了一些方面
    * {none} = 不相关，未能解决查询
- 仅返回数值评分（{full}, {partial}, 或 {none}），无需解释或其他附加文本。
"""
```


```python
def evaluate_response(question, response, true_answer):
        """
        根据忠实度和相关性评估 AI 生成的回答质量

        Args:
        question (str): 用户的原始问题
        response (str): 被评估的 AI 生成的回答
        true_answer (str): 作为基准的真实答案

        Returns:
        Tuple[float, float]: 包含 (忠实度评分, 相关性评分) 的元组。
                             每个评分可能是：1.0（完全匹配）、0.5（部分匹配）或 0.0（无匹配）。
        """
        # 格式化评估提示
        faithfulness_prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
                question=question,
                response=response,
                true_answer=true_answer,
                full=SCORE_FULL,
                partial=SCORE_PARTIAL,
                none=SCORE_NONE
        )

        relevancy_prompt = RELEVANCY_PROMPT_TEMPLATE.format(
                question=question,
                response=response,
                full=SCORE_FULL,
                partial=SCORE_PARTIAL,
                none=SCORE_NONE
        )

        # 模型进行忠实度评估
        faithfulness_response = client.chat.completions.create(
               model=os.getenv("LLM_MODEL_ID"),
                temperature=0,
                messages=[
                        {"role": "system", "content": "你是一个客观的评估者，仅返回数值评分。"},
                        {"role": "user", "content": faithfulness_prompt}
                ]
        )

        # 模型进行相关性评估
        relevancy_response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL_ID"),
                temperature=0,
                messages=[
                        {"role": "system", "content": "你是一个客观的评估者，仅返回数值评分。"},
                        {"role": "user", "content": relevancy_prompt}
                ]
        )

        # 提取评分并处理潜在的解析错误
        try:
                faithfulness_score = float(faithfulness_response.choices[0].message.content.strip())
        except ValueError:
                print("Warning: 无法解析忠实度评分，将默认为 0")
                faithfulness_score = 0.0

        try:
                relevancy_score = float(relevancy_response.choices[0].message.content.strip())
        except ValueError:
                print("Warning: 无法解析相关性评分，将默认为 0")
                relevancy_score = 0.0

        return faithfulness_score, relevancy_score

# 第一条验证数据的真实答案
true_answer = data[3]['ideal_answer']

# 评估块大小为 256 和 128 的回答
faithfulness, relevancy = evaluate_response(query, ai_responses_dict[256], true_answer)
faithfulness2, relevancy2 = evaluate_response(query, ai_responses_dict[128], true_answer)

# 打印评估分数
print(f"忠实度评分 (Chunk Size 256): {faithfulness}")
print(f"相关性评分 (Chunk Size 256): {relevancy}")

print(f"\n")

print(f"忠实度评分 (Chunk Size 128): {faithfulness2}")
print(f"忠实度评分 (Chunk Size 128): {relevancy2}")
```

    忠实度评分 (Chunk Size 256): 1.0
    相关性评分 (Chunk Size 256): 1.0
    
    
    忠实度评分 (Chunk Size 128): 1.0
    忠实度评分 (Chunk Size 128): 1.0

