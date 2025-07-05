# RAG 入门

检索增强生成（RAG）是一种混合方法，它结合了信息检索与生成模型。通过结合外部知识，它增强了语言模型的表现，提高了准确性和事实的正确性。

-----
实现步骤：
- **Data Ingestion（数据采集）**: 加载和预处理文本数据。
- **Chunking（分块处理）**: 将数据分割成更小的块以提高检索性能。
- **Embedding Creation（嵌入创建）**: 使用嵌入模型将文本块转换为数值表示。
- **Semantic Search（语义搜索）**: 根据用户查询检索相关块。
- **Response Generation（响应生成）**：使用语言模型根据检索到的文本生成响应。


## 1.1 设置环境


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



# 1.2 从 PDF 文件中提取文本

使用 PyMuPDF 库从 PDF 文件中提取文本


```python
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file and prints the first `num_chars` characters.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text

    # Iterate through each page in the PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # Get the page
        text = page.get_text("text")  # Extract text from the page
        all_text += text  # Append the extracted text to the all_text string

    return all_text  # Return the extracted text
```

# 1.3 对提取的文本进行分块

将文本切分成更小的、重叠的块以提高检索准确性


```python
def chunk_text(text, n, overlap):
    """
    Chunks the given text into segments of n characters with overlap.

    Args:
    text (str): 文本
    n (int): 块长度
    overlap (int): 重叠度

    Returns:
    List[str]: A list of text chunks.
    """
    chunks = []  # Initialize an empty list to store the chunks

    # Loop through the text with a step size of (n - overlap)
    for i in range(0, len(text), n - overlap):
        # Append a chunk of text from index i to i + n to the chunks list
        chunks.append(text[i:i + n])

    return chunks
```

# 1.4 设置 OpenAI API 客户端

初始化 OpenAI 客户端以生成嵌入和响应


```python
# 国内支持类OpenAI的API都可，我用的是火山引擎的，需要配置对应的base_url和api_key

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY")
)
```

# 1.5 从 PDF 文件中提取和分块文本

加载 PDF，提取文本并将其分割成块


```python
# PDF file
pdf_path = "../../data/AI_Information.en.zh-CN.pdf"

# 提取文本
extracted_text = extract_text_from_pdf(pdf_path)

# 切分文本块，块长度为300，重叠度为50
text_chunks = chunk_text(extracted_text, 500, 100)

# 文本块的数量
print("Number of text chunks:", len(text_chunks))

# 第一个文本块
print("\nFirst text chunk:")
print(text_chunks[0])
```

    Number of text chunks: 26
    
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
    第⼆章：⼈⼯智能


# 1.6 文本块创建嵌入

嵌入将文本转换为数值向量，这允许进行高效的相似性搜索


```python
# from sentence_transformers import SentenceTransformer, util
# from typing import List
# from pathlib import Path
#
#
# def create_embeddings(text: List[str], model_path: str = "../rag_naive/model/gte-base-zh") -> List[List[float]]:
#     """
#     Creates embeddings for the given text using the local-embedding model.
#     eg: modelscope gte-base-zh
#     """
#     # Create embeddings for the input text using the specified model
#
#     st_model = SentenceTransformer(model_name_or_path=model_path)
#     st_embeddings = st_model.encode(text, normalize_embeddings=True)
#     response = [embedding.tolist() for embedding in st_embeddings]
#
#     return response

def create_embeddings(text):
    # Create embeddings for the input text using the specified model
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL_ID"),
        input=text
    )

    return response  # Return the response containing the embeddings


# 文本块的嵌入向量
response = create_embeddings(text_chunks)

```

# 1.7 语义搜索

实现余弦相似度来找到与用户查询最相关的文本片段


```python
def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    # Compute the dot product of the two vectors and divide by the product of their norms
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```


```python
def semantic_search(query, text_chunks, embeddings, k=5):
    """
    Performs semantic search on the text chunks using the given query and embeddings.

    Args:
    query (str): The query for the semantic search.
    text_chunks (List[str]): A list of text chunks to search through.
    embeddings (List[dict]): A list of embeddings for the text chunks.
    k (int): The number of top relevant text chunks to return. Default is 5.

    Returns:
    List[str]: A list of the top k most relevant text chunks based on the query.
    """
    # Create an embedding for the query
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []  # Initialize a list to store similarity scores

    # Calculate similarity scores between the query embedding and each text chunk embedding
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))  # Append the index and similarity score

    # Sort the similarity scores in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # Get the indices of the top k most similar text chunks
    top_indices = [index for index, _ in similarity_scores[:k]]
    # Return the top k most relevant text chunks
    return [text_chunks[index] for index in top_indices]
```

# 1.8 在提取的文本块上进行语义搜索




```python
# Load the validation data from a JSON file
with open('../../data/val.json', encoding="utf-8") as f:
    data = json.load(f)

# Extract the first query from the validation data
query = data[0]['question']

# Perform semantic search to find the top 2 most relevant text chunks for the query
top_chunks = semantic_search(query, text_chunks, response.data, k=2)

# Print the query
print("Query:", query)

# Print the top 2 most relevant text chunks
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")
```

    Query: 什么是‘可解释人工智能’，为什么它被认为很重要？
    Context 1:
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
    能以及解决潜
    =====================================
    Context 2:
    。让⼈⼯智能系统易于理解，并深⼊了解其决策过
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
    负责任地开发和部署⼈⼯智能对于确保其效益得到⼴泛共享并降低其⻛险⾄关重要。这涉及遵守
    =====================================


# 1.9 基于检索到的片段生成响应


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

    **可解释人工智能（XAI）**是指那些能够使其决策过程和结果易于理解和解释的人工智能系统。它旨在让用户、开发者和其他利益相关者能够清晰地了解AI系统是如何做出特定决策的。
    
    **为什么可解释人工智能很重要？**
    
    1. **建立信任**：透明度和可解释性是建立对人工智能系统信任的关键。当用户能够理解AI的决策过程时，他们更有可能信任其可靠性和公平性。
    
    2. **评估公平性和准确性**：可解释人工智能使用户能够评估系统的决策是否公平、准确，从而避免潜在的偏见和歧视。
    
    3. **解决偏见**：AI系统可能会继承并放大其训练数据中存在的偏见。可解释性有助于识别和纠正这些偏见，确保结果更加公正。
    
    4. **问责与责任**：明确AI系统的决策过程有助于建立问责制和责任制，使开发者、部署者和用户都能明确各自的角色和职责。
    
    5. **遵守法规**：在某些领域，法律和法规要求决策过程必须是透明的。可解释人工智能有助于满足这些要求。
    
    综上所述，可解释人工智能对于确保AI系统的透明度、公平性、可靠性和合规性至关重要，从而促进其广泛接受和信任。


# 1.10 评估响应质量




```python
# Define the system prompt for the evaluation system
evaluate_system_prompt = "你是一个智能评估系统，负责评估AI助手的回答。如果AI助手的回答与真实答案非常接近，则评分为1。如果回答错误或与真实答案不符，则评分为0。如果回答部分符合真实答案，则评分为0.5。"

# Create the evaluation prompt by combining the user query, AI response, true response, and evaluation system prompt
evaluation_prompt = f"用户问题: {query}\nAI回答:\n{ai_response}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# Generate the evaluation response using the evaluation system prompt and evaluation prompt
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)
print(evaluation_response)
```

    根据提供的用户问题、AI回答和真实答案，AI助手的回答与真实答案非常接近，涵盖了可解释人工智能（XAI）的定义及其重要性的多个方面，包括建立信任、评估公平性和准确性、解决偏见、问责与责任以及遵守法规。这些内容与真实答案中提到的建立信任、问责制和确保公平性高度一致。
    
    因此，AI助手的回答可以被认为是全面且准确的，与真实答案非常接近。
    
    **评分：1**

