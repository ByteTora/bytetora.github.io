"""
chunk不同大小 核心函数
"""
import fitz


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


if __name__ == '__main__':

    pdf_path = "../../data/AI_Information.en.zh-CN.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    # 定义要评估的不同块大小
    chunk_sizes = [128, 256, 512]

    # 创建一个字典，用于存储每个块大小对应的文本块
    text_chunks_dict = {size: chunk_text(extracted_text, size, size // 5) for size in chunk_sizes}

    # 打印每个块大小生成的块数量
    for size, chunks in text_chunks_dict.items():
        print(f"Chunk Size: {size}, Number of Chunks: {len(chunks)}")
