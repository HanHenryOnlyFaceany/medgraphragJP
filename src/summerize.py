from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import os

"""
医疗文本结构化摘要生成模块

这个模块主要用于将医疗相关文本进行分块处理，并为每个块生成结构化摘要。
摘要包含解剖结构、身体功能、测量数据、实验室数据、药物信息等多个医疗相关类别。
"""

# OpenAI API配置
openai_api_key = os.getenv("OPENAI_API_KEY")

# 系统提示词：定义了结构化摘要的格式和所需包含的医疗信息类别
sum_prompt = """
Generate a structured summary from the provided medical source (report, paper, or book), strictly adhering to the following categories. The summary should list key information under each category in a concise format: 'CATEGORY_NAME: Key information'. No additional explanations or detailed descriptions are necessary unless directly related to the categories:

ANATOMICAL_STRUCTURE: Mention any anatomical structures specifically discussed.
BODY_FUNCTION: List any body functions highlighted.
BODY_MEASUREMENT: Include normal measurements like blood pressure or temperature.
BM_RESULT: Results of these measurements.
BM_UNIT: Units for each measurement.
BM_VALUE: Values of these measurements.
LABORATORY_DATA: Outline any laboratory tests mentioned.
LAB_RESULT: Outcomes of these tests (e.g., 'increased', 'decreased').
LAB_VALUE: Specific values from the tests.
LAB_UNIT: Units of measurement for these values.
MEDICINE: Name medications discussed.
MED_DOSE, MED_DURATION, MED_FORM, MED_FREQUENCY, MED_ROUTE, MED_STATUS, MED_STRENGTH, MED_UNIT, MED_TOTALDOSE: Provide concise details for each medication attribute.
PROBLEM: Identify any medical conditions or findings.
PROCEDURE: Describe any procedures.
PROCEDURE_RESULT: Outcomes of these procedures.
PROC_METHOD: Methods used.
SEVERITY: Severity of the conditions mentioned.
MEDICAL_DEVICE: List any medical devices used.
SUBSTANCE_ABUSE: Note any substance abuse mentioned.
Each category should be addressed only if relevant to the content of the medical source. Ensure the summary is clear and direct, suitable for quick reference.
"""

def call_openai_api(chunk):
    """
    调用OpenAI API生成文本摘要
    
    使用PPIO的API服务调用大语言模型，将输入的文本块转换为结构化摘要。
    
    Args:
        chunk (str): 需要处理的文本块
        
    Returns:
        str: 生成的结构化摘要内容
    """
    client = OpenAI(
        api_key = "sk_J7l6-K7MQB_9aPomZCuXWXrmxEUF_U91rXvGfRypmj0",
        base_url = "https://api.ppinfra.com/v3/openai",
    )
    
    response = client.chat.completions.create(
        model="qwen/qwen-2.5-72b-instruct",
        messages=[
            {"role": "system", "content": sum_prompt},
            {"role": "user", "content": f" {chunk}"},
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

def split_into_chunks(text, tokens=500):
    """
    将长文本分割成较小的块
    
    使用tiktoken库将文本按照指定的token数量分割成多个小块，
    确保每个块的大小适合模型处理。
    
    Args:
        text (str): 需要分割的原始文本
        tokens (int): 每个块的最大token数量，默认500
        
    Returns:
        list: 分割后的文本块列表
    """
    try:
        # 直接使用 tiktoken.get_encoding 获取编码器
        encoding = tiktoken.get_encoding("cl100k_base")
        words = encoding.encode(text)
        chunks = []
        for i in range(0, len(words), tokens):
            chunks.append(' '.join(encoding.decode(words[i:i + tokens])))
        return chunks
    except Exception as e:
        print(f"分块过程出错: {str(e)}")
        # 如果出错，返回原文本作为单个块
        return [text]

def process_chunks(content):
    """
    处理文本内容并生成摘要
    
    将输入的文本内容分块，并使用多线程并行处理每个块，
    生成对应的结构化摘要。
    
    Args:
        content (str): 需要处理的原始文本内容
        
    Returns:
        list: 所有文本块的摘要列表
    """
    chunks = split_into_chunks(content)

    # 使用线程池并行处理多个文本块
    with ThreadPoolExecutor() as executor:
        responses = list(executor.map(call_openai_api, chunks))
    return responses


if __name__ == "__main__":
    # 测试代码
    content = " 这位76岁的患者的血压是否有异常？"
    process_chunks(content)

# 处理时间可能较长，取决于输入数据的大小