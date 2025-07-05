# 语义块切分 Semantic Chunking

文本块切分是检索增强生成（RAG）中的关键步骤，其中将大文本体分割成有意义的段落以提高检索准确性。与固定长度块切分不同，语义块切分是根据句子之间的内容相似性来分割文本的。

------
分块断点方法：
- **Percentile（百分位）**: 找到所有相似度差异小于设定的百分点的内容，并在超过此值的下降处分割块。
- **Standard Deviation（标准差）**: 相似度超过 X 个标准差以下的分割点。
- **Interquartile Range (IQR)（四分位距）**: 使用四分位距（Q3 - Q1）来确定分割点。

------
实现步骤：
- 从PDF文件中提取文本：按句子进行切分
- 提取的文本创建语义分块：
    - 将前后两个相邻的句子计算相似度
    - 根据相似度下降计算分块的断点，断点方法有三种：百分位、标准差和四分位距
    - 然后根据断点分割文本，得到语义块
- 创建嵌入
- 根据查询并检索文档
- 将检索出来的文本用于模型生成回答

# 设置环境


```python
import fitz
import os
import numpy as np
import json
from openai import OpenAI
from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer, util
# from typing import List

load_dotenv()
```




    True



# 从 PDF 文件中提取文本


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


# 设置 OpenAI API 客户端

初始化 OpenAI 客户端以生成嵌入和响应


```python
# 国内支持类OpenAI的API都可，需要配置对应的base_url和api_key

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
```

# 创建句子级嵌入(sentence-level embedding)

将文本分割成句子并生成嵌入


```python
def get_embedding(text):
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL_ID"),
        input=text
    )
    return np.array(response.data[0].embedding)

# Splitting text into sentences (basic split)
sentences = extracted_text.split("。")
print(len(sentences))
# Generate embeddings for each sentence
# FIXME: 因为PDF切分的最后一个句子为空，需要过滤掉
embeddings = [get_embedding(sentence) for sentence in sentences if sentence]
print(f"Generated {len(embeddings)} sentence embeddings.")

# def get_embedding(text, model_path: str = "../rag_naive/model/gte-base-zh"):
#     """
#     Creates an embedding for the given text using OpenAI.
#
#     Args:
#     text (str): Input text.
#     model (str): Embedding model name.
#
#     Returns:
#     np.ndarray: The embedding vector.
#     """
#     st_model = SentenceTransformer(model_name_or_path=model_path)
#     st_embeddings = st_model.encode(text, normalize_embeddings=True)
#     response = [embedding.tolist() for embedding in st_embeddings]
#
#     return np.array(response)
#
# # Splitting text into sentences (basic split)
# sentences = extracted_text.split("。")
# print(len(sentences))
#
# # Generate embeddings for each sentence
# embeddings = [get_embedding(sentence) for sentence in sentences]
#
# print(f"Generated {len(embeddings)} sentence embeddings.")
```

    257
    Generated 256 sentence embeddings.


# 计算相似度差异

计算连续句子的余弦相似度


```python
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): First vector.
    vec2 (np.ndarray): Second vector.

    Returns:
    float: Cosine similarity.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compute similarity between consecutive sentences
similarities = [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
```

# 实现语义分块 Semantic Chunking

实现了三种不同的方法来查找断点


```python
def compute_breakpoints(similarities, method="percentile", threshold=90):
    """
    根据相似度下降计算分块的断点。

    Args:
        similarities (List[float]): 句子之间的相似度分数列表。
        method (str): 'percentile'（百分位）、'standard_deviation'（标准差）或 'interquartile'（四分位距）。
        threshold (float): 阈值（对于 'percentile' 是百分位数，对于 'standard_deviation' 是标准差倍数）。

    Returns:
        List[int]: 分块的索引列表。
    """
    # 根据选定的方法确定阈值
    if method == "percentile":
        # 计算相似度分数的第 X 百分位数
        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        # 计算相似度分数的均值和标准差。
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        # 将阈值设置为均值减去 X 倍的标准差
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        # 计算第一和第三四分位数（Q1 和 Q3）。
        q1, q3 = np.percentile(similarities, [25, 75])
        # 使用 IQR 规则（四分位距规则）设置阈值
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        # 如果提供了无效的方法，则抛出异常
        raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

    # 找出相似度低于阈值的索引
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]

# 使用百分位法计算断点，阈值为90
breakpoints = compute_breakpoints(similarities, method="percentile", threshold=90)
breakpoints
```




    [0,
     1,
     2,
     3,
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15,
     16,
     17,
     18,
     19,
     20,
     21,
     22,
     23,
     24,
     25,
     26,
     27,
     28,
     29,
     30,
     32,
     33,
     34,
     35,
     36,
     37,
     38,
     39,
     40,
     41,
     42,
     44,
     45,
     46,
     48,
     49,
     50,
     52,
     53,
     54,
     55,
     56,
     57,
     58,
     59,
     60,
     61,
     62,
     63,
     65,
     66,
     67,
     68,
     69,
     70,
     71,
     72,
     73,
     74,
     76,
     77,
     78,
     79,
     81,
     82,
     83,
     84,
     85,
     86,
     87,
     88,
     90,
     91,
     92,
     93,
     94,
     95,
     96,
     97,
     98,
     99,
     100,
     101,
     102,
     103,
     105,
     106,
     107,
     109,
     110,
     111,
     112,
     113,
     114,
     115,
     116,
     117,
     118,
     119,
     120,
     121,
     122,
     124,
     125,
     127,
     129,
     130,
     131,
     132,
     133,
     134,
     135,
     136,
     137,
     138,
     139,
     140,
     141,
     142,
     143,
     144,
     145,
     147,
     148,
     149,
     150,
     151,
     152,
     154,
     155,
     157,
     159,
     161,
     162,
     163,
     164,
     165,
     166,
     167,
     168,
     169,
     170,
     171,
     172,
     173,
     174,
     175,
     176,
     177,
     179,
     180,
     181,
     182,
     183,
     184,
     185,
     186,
     187,
     188,
     189,
     190,
     191,
     192,
     193,
     194,
     195,
     197,
     199,
     200,
     201,
     202,
     203,
     204,
     205,
     206,
     207,
     208,
     209,
     211,
     212,
     213,
     214,
     215,
     216,
     217,
     218,
     219,
     220,
     221,
     223,
     224,
     225,
     226,
     227,
     228,
     229,
     230,
     231,
     232,
     233,
     234,
     236,
     237,
     238,
     239,
     240,
     241,
     243,
     244,
     245,
     246,
     247,
     248,
     249,
     250,
     251,
     252,
     253]



# 将文本分割成语义块

将文本基于断点进行分割


```python
def split_into_chunks(sentences, breakpoints):
    """
    将句子分割为语义块

    Args:
    sentences (List[str]): 句子列表
    breakpoints (List[int]): 进行分块的索引位置

    Returns:
    List[str]: 文本块列表
    """
    chunks = []  # Initialize an empty list to store the chunks
    start = 0  # Initialize the start index

    # 遍历每个断点以创建块
    for bp in breakpoints:
        # 将从起始位置到当前断点的句子块追加到列表中
        chunks.append("。".join(sentences[start:bp + 1]) + "。")
        start = bp + 1  # 将起始索引更新为断点后的下一个句子

    # 将剩余的句子作为最后一个块追加
    chunks.append("。".join(sentences[start:]))
    return chunks  # Return the list of chunks

# split_into_chunks 函数创建文本块
text_chunks = split_into_chunks(sentences, breakpoints)

# Print the number of chunks created
print(f"Number of semantic chunks: {len(text_chunks)}")

# Print the first chunk to verify the result
print("\nFirst text chunk:")
print(text_chunks[0])
```

    Number of semantic chunks: 230
    
    First text chunk:
    理解⼈⼯智能
    第⼀章：⼈⼯智能简介
    ⼈⼯智能 (AI) 是指数字计算机或计算机控制的机器⼈执⾏通常与智能⽣物相关的任务的能⼒。


# 语义块创建嵌入

为每个片段创建嵌入，以便后续检索


```python
def create_embeddings(text_chunks):
    """
    Creates embeddings for each text chunk.

    Args:
    text_chunks (List[str]): List of text chunks.

    Returns:
    List[np.ndarray]: List of embedding vectors.
    """
    # Generate embeddings for each text chunk using the get_embedding function
    return [get_embedding(chunk) for chunk in text_chunks]

# Create chunk embeddings using the create_embeddings function
chunk_embeddings = create_embeddings(text_chunks)
```

# 语义搜索

余弦相似度来检索最相关的片段


```python
def semantic_search(query, text_chunks, chunk_embeddings, k=5):
    """
    查询找到最相关的文本块

    Args:
    query (str): Search query.
    text_chunks (List[str]): List of text chunks.
    chunk_embeddings (List[np.ndarray]): List of chunk embeddings.
    k (int): Number of top results to return.

    Returns:
    List[str]: Top-k relevant chunks.
    """
    # 为查询生成嵌入
    query_embedding = get_embedding(query)

    # 计算查询嵌入与每个块嵌入之间的余弦相似度
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]

    # 获取最相似的 k 个块的索引
    top_indices = np.argsort(similarities)[-k:][::-1]

    # 返回最相关的 k 个文本块
    return [text_chunks[i] for i in top_indices]
```


```python
# Load the validation data from a JSON file
with open('../../data/val.json', encoding="utf-8") as f:
    data = json.load(f)

# Extract the first query from the validation data
query = data[0]['question']

# Get top 2 relevant chunks
top_chunks = semantic_search(query, text_chunks, chunk_embeddings, k=2)

# Print the query
print(f"Query: {query}")

# Print the top 2 most relevant text chunks
for i, chunk in enumerate(top_chunks):
    print(f"Context {i+1}:\n{chunk}\n{'='*40}")
```

    Query: 什么是‘可解释人工智能’，为什么它被认为很重要？
    Context 1:
    
    透明度和可解释性
    透明度和可解释性对于建⽴对⼈⼯智能系统的信任⾄关重要。
    ========================================
    Context 2:
    
    透明度和可解释性
    许多⼈⼯智能系统，尤其是深度学习模型，都是“⿊匣⼦”，很难理解它们是如何做出决策的。
    ========================================


# 基于检索到的片段生成响应



```python
# Define the system prompt for the AI assistant
system_prompt = "你是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"

def generate_response(system_prompt, user_message):
    """
    Generates a response from the AI model based on the system prompt and user message.

    Args:
    system_prompt (str): The system prompt to guide the AI's behavior.
    user_message (str): The user's message or query.

    Returns:
    dict: The response from the AI model.
    """

    response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL_ID"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            top_p=0.8,
            presence_penalty=1.05,
            max_tokens=4096,
        )
    return response.choices[0].message.content

# Create the user prompt based on the top chunks
user_prompt = "\n".join([f"上下文内容 {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\n问题: {query}"

# Generate AI response
ai_response = generate_response(system_prompt, user_prompt)
print(ai_response)
```

    '可解释人工智能'（Explainable AI, XAI）指的是那些能够向用户清晰地解释其决策过程和逻辑的人工智能系统。这种解释可以是关于模型如何处理输入数据、做出特定决策的原因，以及这些决策背后的推理过程。
    
    它被认为很重要，主要有以下几个原因：
    
    1. **建立信任**：当用户能够理解AI系统是如何工作时，他们更有可能信任这些系统的决策。如上下文内容1所述，透明度和可解释性对于建立对人工智能系统的信任至关重要。
    
    2. **安全性**：了解AI的决策过程有助于识别和纠正潜在的错误或偏差，从而提高系统的安全性和可靠性。
    
    3. **合规性**：在某些行业（如医疗、金融等），法规可能要求AI系统提供决策的解释，以确保其符合法律和伦理标准。
    
    4. **改进和优化**：通过理解AI的决策逻辑，开发者可以更好地优化模型，提高其性能和准确性。
    
    5. **用户接受度**：用户更愿意接受那些他们能够理解和控制的AI系统，这有助于推广AI技术的应用。
    
    综上所述，可解释人工智能对于提升AI系统的透明度、信任度、安全性和合规性等方面都具有重要意义。


# 评估人工智能响应


```python
# Define the system prompt for the evaluation system
evaluate_system_prompt = "你是一个智能评估系统，负责评估AI助手的回答。如果AI助手的回答与真实答案非常接近，则评分为1。如果回答错误或与真实答案不符，则评分为0。如果回答部分符合真实答案，则评分为0.5。"

# Create the evaluation prompt by combining the user query, AI response, true response, and evaluation system prompt
evaluation_prompt = f"用户问题: {query}\nAI回答:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# Generate the evaluation response using the evaluation system prompt and evaluation prompt
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print(evaluation_response)
```

    根据提供的AI回答和真实答案，AI助手的回答与真实答案非常接近，涵盖了可解释人工智能（XAI）的定义、重要性及其多个方面的应用。具体来说：
    
    1. **定义**：AI回答正确地解释了XAI是能够清晰解释决策过程和逻辑的人工智能系统。
    2. **重要性**：
       - **建立信任**：AI回答提到了透明度和可解释性对于建立信任的重要性，与真实答案一致。
       - **安全性**：AI回答提到了通过了解决策过程来提高系统的安全性和可靠性，这与真实答案中的问责制有一定关联。
       - **合规性**：AI回答提到了法规要求解释决策，确保符合法律和伦理标准，这与真实答案中的公平性有一定关联。
       - **改进和优化**：AI回答提到了通过理解决策逻辑来优化模型，这是对XAI重要性的补充说明。
       - **用户接受度**：AI回答提到了用户更愿意接受可理解的AI系统，这与建立信任相关。
    
    虽然AI回答没有直接提到“问责制”和“公平性”，但其提到的多个方面（如安全性、合规性）与这些概念有间接关联，且整体上涵盖了真实答案的核心要点。
    
    因此，AI助手的回答与真实答案非常接近，评分为1。

