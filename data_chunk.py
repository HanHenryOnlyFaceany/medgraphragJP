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
        
        essay_propositions.extend(propositions)
        print (f"Done with {i}")

    ac = AgenticChunker()
    ac.add_propositions(essay_propositions)
    ac.pretty_print_chunks()
    chunks = ac.get_chunks(get_type='list_of_strings')

    return chunks
    print(chunks)