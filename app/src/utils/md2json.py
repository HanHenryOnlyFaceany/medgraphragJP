import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import tiktoken
import os
import json

"""
医疗文本结构化摘要生成模块

这个模块主要用于将医疗相关文本进行分块处理，并为每个块生成结构化摘要。
摘要包含解剖结构、身体功能、测量数据、实验室数据、药物信息等多个医疗相关类别。
"""

# OpenAI API配置
openai_api_key = os.getenv("OPENAI_API_KEY")

# 系统提示词：定义了结构化文本的格式
system_prompt = """
你必须严格按照以下JSON格式输出，不要添加任何额外的文本、解释或注释：

[
  {"type": "SECTION_1", "title": "标题内容", "section_title": "一级标题内容"},
  {"type": "SECTION_2", "title": "标题内容", "section_title": "二级标题内容"},
  {"type": "ABSTRACT", "content": "摘要内容", "section_title": "所属标题"},
  {"type": "KEYWORDS", "content": "关键词内容", "section_title": "所属标题"},
  {"type": "PARAGRAPH", "content": "段落内容", "section_title": "所属标题"},
  {"type": "TABLE", "content": "表格内容", "section_title": "表格名称"}
]

处理要求：
1. 仅保留与三元组抽取相关的中文内容，包括章节标题、中文摘要、中文关键词、正文段落和表格内容
2. 忽略图片,英文摘要等无关内容，但必须提取表格信息
3. 对每一部分内容加上结构化标签，标签类型包括：
   - SECTION_1, SECTION_2, ...（章节标题，数字代表层级）
   - ABSTRACT（中文摘要）
   - KEYWORDS（中文关键词）
   - PARAGRAPH（正文段落）
   - TABLE（表格内容）
4. 严格区分标题和内容，不要将它们合并在一起
5. 如果一行文本以数字编号或特殊符号开头（如"1."、"•"等），且后面跟随完整句子，则视为段落内容而非标题
6. 标题通常较短，不包含完整的句子结构，而段落内容通常较长且包含完整句子
7. 确保每个PARAGRAPH和TABLE对象的section_title字段正确关联到其所属的标题
8. 对于每个标题下的正文内容，按照标题进行分段，同一个标题下的段落单独保存为一个JSON对象
9. 每个JSON对象必须包含"type"字段和以下字段之一：
   - 对于标题：使用"title"字段
   - 对于内容：使用"content"字段
10. 所有JSON对象都必须包含"section_title"字段，表示所属的章节标题
11. 只保留包含中文字符的内容，去除所有英文内容和无关信息
12. 对于表格内容：
    - 识别Markdown中的表格语法（|---|---|形式）或其他表格标记
    - 将表格标题作为section_title字段的值
    - 将表格内容转换为结构化文本，保留行列关系
    - 如果表格有标题行，将其作为列名保留
    - 对于复杂表格（如合并单元格），尽量保持原始结构
    - 对于医学指标、药物剂量、治疗方案等表格，确保数值和单位正确关联

重要提示：
- 你的输出必须是一个有效的JSON数组，不要添加任何额外的文本
- 不要在JSON前后添加```json或其他标记
- 确保所有字符串都使用双引号，并正确转义内部的双引号
- 不要使用单引号或其他非标准JSON语法
- 不要添加注释或解释
- 确保输出的JSON可以被JSON.parse()直接解析

请处理以下Markdown文本内容：
"""

structured_prompt_first = """
你是一名医学知识工程师。请对以下医学指南或论文的Markdown文本内容进行结构化分段和标签化处理，本次内容为文档的开始部分，请必须输出标题（TITLE）、摘要（ABSTRACT）和关键词（KEYWORDS）目的是为后续医学知识图谱的三元组抽取做准备。
"""
structured_prompt_first = structured_prompt_first + system_prompt


structured_prompt_other = """
你是一名医学知识工程师。请对以下医学指南或论文的Markdown文本内容进行结构化分段和标签化处理，目的是为后续医学知识图谱的三元组抽取做准备。
本次内容为文档中间部分，请只输出章节标题和正文段落以及表格（SECTION_x、PARAGRAPH、TABLE），不要输出摘要（ABSTRACT）和关键词（KEYWORDS）。
"""
structured_prompt_other = structured_prompt_other + system_prompt

def call_openai_api(chunk, prompt=structured_prompt_first):
    """
    调用OpenAI API生成文本摘要
    
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
            {"role": "system", "content": prompt},
            {"role": "user", "content": f" {chunk}"},
        ],
        temperature=0.3,
        max_tokens=10000,
    )
    return response.choices[0].message.content

def remove_images_and_english_abstract_from_md(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 去除所有Markdown图片语法 ![xxx](xxx)
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    # 去除英文摘要（如【Abstract】开头到下一个空行或段落）
    content = re.sub(r'【Abstract】.*?(?=\n\s*\n|$)', '', content, flags=re.DOTALL)
    # 去除英文关键词（如【Key words】开头到下一个空行或段落）
    content = re.sub(r'【Key words】.*?(?=\n\s*\n|$)', '', content, flags=re.DOTALL)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

def process_md_to_json(input_path, output_json_path):
    """
    处理Markdown文件，调用API生成结构化JSON，并保存为文件
    
    Args:
        input_path (str): 输入Markdown文件路径
        output_json_path (str): 输出JSON文件路径
    """
    # 读取Markdown文件
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 调用API生成结构化JSON
    json_content = call_openai_api(content)
    
    # 尝试解析JSON以确保格式正确
    try:
        parsed_json = json.loads(json_content)
        # 将解析后的JSON重新格式化并保存
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_json, f, ensure_ascii=False, indent=2)
        print(f"结构化处理完成，JSON已保存为: {output_json_path}")
    except json.JSONDecodeError as e:
        # 如果API返回的不是有效JSON，则直接保存原始内容
        with open(output_json_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        print(f"警告: API返回的内容不是有效JSON格式，已保存原始内容: {output_json_path}")
        print(f"错误信息: {str(e)}")

def split_text_by_paragraphs(text, max_paragraphs=8):
    """
    按段落分块，每块最多包含max_paragraphs个段落
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    print(f"文本段落数: {len(paragraphs)}")
    for i in range(0, len(paragraphs), max_paragraphs):
        chunk = '\n\n'.join(paragraphs[i:i+max_paragraphs])
        chunks.append(chunk)
    return chunks

def process_md_to_json_with_chunks(input_path, output_json_path, max_paragraphs=8):
    """
    分块处理Markdown文件，每块调用API，合并所有JSON结果
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 去除所有Markdown图片语法 ![xxx](xxx)
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    # 去除英文摘要（如【Abstract】开头到下一个空行或段落）
    content = re.sub(r'【Abstract】.*?(?=\n\s*\n|$)', '', content, flags=re.DOTALL)
    # 去除英文关键词（如【Key words】开头到下一个空行或段落）
    content = re.sub(r'【Key words】.*?(?=\n\s*\n|$)', '', content, flags=re.DOTALL)
    chunks = split_text_by_paragraphs(content, max_paragraphs=max_paragraphs)
    all_results = []
    for idx, chunk in enumerate(chunks):
        print(f"正在处理第{idx+1}块/共{len(chunks)}块...")
        if idx+1 == 1:
            print(structured_prompt_first)
            json_content = call_openai_api(chunk,prompt=structured_prompt_first)
        else:
            json_content = call_openai_api(chunk, prompt=structured_prompt_other)
        try:
            parsed_json = json.loads(json_content)
            if isinstance(parsed_json, list):
                all_results.extend(parsed_json)
            else:
                all_results.append(parsed_json)
        except json.JSONDecodeError as e:
            print(f"第{idx+1}块解析失败，原始内容已跳过。错误信息: {str(e)}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"全部分块处理完成，结构化JSON已保存为: {output_json_path}")

if __name__ == '__main__':
    input_md = '/Users/hanhenry99/jianpei/medgraphragJP/data/md/医学指南下载_2025320/糖尿病/糖尿病肾病诊治专家共识解读_1745206973.007173.md'
    
    # # 1. 首先去除图片和英文摘要
    # output_md = input_md.replace('.md', '_noimg.md')
    # remove_images_and_english_abstract_from_md(input_md, output_md)
    # print("去除图片和英文摘要完成，结果已保存为:", output_md)
    
    # 2. 处理清理后的MD文件，生成结构化JSON
    output_json = input_md.replace('.md', '_structured.json')
    process_md_to_json_with_chunks(output_md, output_json)
    

