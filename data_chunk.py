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

# Pydantic data class
class Sentences(BaseModel):
    sentences: List[str]


def get_propositions(text, runnable, extraction_chain):
    runnable_output = runnable.invoke({
    	"input": text
    }).content
    
    # 修改这行，直接使用返回的 Sentences 对象
    propositions = extraction_chain.invoke(runnable_output).sentences
    return propositions
"""
- 文章分段
- 每段提取关键句子
- 汇总所有句子
- 使用 AgenticChunker 进行最终处理
- 返回处理后的文本块
"""
def run_chunk(essay):

    obj = hub.pull("wfh/proposal-indexing")
    llm = init_chat_model(            
        model="qwen/qwen-2.5-72b-instruct", 
        model_provider="openai",
        api_key="sk_J7l6-K7MQB_9aPomZCuXWXrmxEUF_U91rXvGfRypmj0",
        base_url="https://api.ppinfra.com/v3/openai"
    )

    runnable = obj | llm

    # Extraction
    # extraction_chain = create_extraction_chain_pydantic(pydantic_schema=Sentences, llm=llm)
    # Extraction
    extraction_chain = llm.with_structured_output(Sentences)


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
    chunks = ac.get_chunks(get_type='list_of_strings')

    return chunks
    print(chunks)