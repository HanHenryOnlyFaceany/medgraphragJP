
import os

def load_high(datapath):
    all_content = ""  # Initialize an empty string to hold all the content
    with open(datapath, 'r', encoding='utf-8') as file:
        for line in file:
            all_content += line.strip() + "\n"  # Append each line to the string, add newline character if needed
    return all_content


def load_markdown(datapath):
    """
    加载Markdown文件内容
    Args:
        datapath: Markdown文件路径
    Returns:
        str: Markdown文件内容
    """
    return load_high(datapath)  # 对于简单的读取，可以直接使用load_high函数





