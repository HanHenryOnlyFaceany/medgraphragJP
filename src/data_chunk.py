from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_community.chat_models import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain.chat_models import init_chat_model
# from langchain_community.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.chains import create_extraction_chain
from typing import Optional, List
from langchain.chains import create_extraction_chain_pydantic
from pydantic import BaseModel, Field
from langchain import hub
import os
from dataloader import load_high
from agentic_chunker import AgenticChunker
from langchain_core.output_parsers import PydanticOutputParser

# Pydantic data class
class Sentences(BaseModel):
    sentences: List[str]

# 设置解析器
parser = PydanticOutputParser(pydantic_object=Sentences)

def get_propositions(text, runnable, extraction_chain):
    # 获取runnable的输出，runnable已经包含了parser，所以直接返回Sentences对象
    runnable_output = runnable.invoke({
        "input": text
    })
    
    # runnable_output已经是Sentences对象，直接获取sentences属性
    return runnable_output.sentences
"""
- 文章分段
- 每段提取关键句子
- 汇总所有句子
- 使用 AgenticChunker 进行最终处理
- 返回处理后的文本块
"""
def run_chunk(essay, gid):

    obj = hub.pull("wfh/proposal-indexing")


    llm = init_chat_model(            
        model="qwen2.5-32b-instruct", 
        model_provider="openai",
        api_key="sk-cJBVEQFNNxphN4Et84F5A9C0083348C6873443A27a18C3De",
        base_url="http://ai.medical-deep.com:20240/v1"
    )
    obj = ChatPromptTemplate.from_messages(
        [
            ("system", """
            将以下input分解为清晰简单的陈述句，确保这些句子在脱离上下文的情况下也能被理解。

            指南:
            1. 将复合句分解为简单句，尽可能保持原文的表述方式
            2. 对于包含附加描述信息的命名实体，将这些信息分离为独立的陈述句
            3. 通过添加必要的修饰词或替换代词（如"它"、"他"、"她"、"他们"、"这个"、"那个"）来使句子去上下文化
            4. 输出格式必须是包含sentences字段的JSON对象

            输入示列："1678年，医学教授乔治·弗兰克·冯·弗兰克瑙（Georg Franck von Franckenau）在德国西南部记录了复活节野兔（Osterhase）的最早证据，但直到18世纪，德国其他地区都知道它的存在。"

            输出示例: {{
                "sentences": [
                    "1678年，Georg Franck von Franckenau在德国西南部首次记录了复活节兔子的最早证据。",
                    "Georg Franck von Franckenau是一位医学教授。",
                    "直到18世纪，复活节兔子的证据在德国其他地区仍然不为人知。"
        ]
}}

{format_instructions}
            """),
            ("human", "请分析以下文本：{input}"),
        ]
    ).partial(
        format_instructions=parser.get_format_instructions()
    )
    runnable = obj | llm | parser

    # Extraction
    # extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)
    # Extraction
    extraction_chain = llm.with_structured_output(Sentences)
    # Extraction
    # extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)


    paragraphs = essay.split("\n\n")

    essay_propositions = []
    """
    每段提取关键句子
    """
    for i, para in enumerate(paragraphs):
        propositions = get_propositions(para, runnable, extraction_chain)
        """
        - append() 会把整个对象作为一个元素添加到列表
        - extend() 会把可迭代对象中的元素逐个展开添加到列表
        """
        essay_propositions.extend(propositions)
        print (f"Done with {i}")
    """
    - agent创建名为chunks的空字典存储所有文本块

    {
        "12345": {
            "chunk_id": "12345",
            "propositions": [
                "The month is October.",
                "The year is 2023."
            ],
            "title": "Date & Time",
            "summary": "This chunk contains information about dates and times, including the current month and year.",
        },
        "67890": {
            "chunk_id": "67890",
            "propositions": [
                "One of the most important things that I didn't understand about the world as a child was the degree to which the returns for performance are superlinear.",
                "Teachers and coaches implicitly told us that the returns were linear.",
                "I heard a thousand times that 'You get out what you put in.'"
            ],
            "title": "Performance Returns",
            "summary": "This chunk contains information about performance returns and how they are perceived differently from reality.",
        }
    }

    """
    ac = AgenticChunker()
    ac.add_propositions(essay_propositions)
    ac.pretty_print_chunks()
    chunks = ac.get_chunks(get_type='list_of_strings', chunks_gid=gid)

    return chunks
    # print(chunks)