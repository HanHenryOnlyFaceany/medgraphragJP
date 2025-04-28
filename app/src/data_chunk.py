from curses import noecho
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
# from src.dataloader import load_high
# from src.agentic_chunker import AgenticChunker
from langchain_core.output_parsers import PydanticOutputParser
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import re

logger = logging.getLogger(__name__)

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
            3. 完全消除指代词，不要使用"它"、"他"、"她"、"他们"、"这个"、"那个"等代词，而是重复使用具体的名词
            4. 使用标题中的主题词作为句子的主语，确保每个句子都有明确的主语
            5. 如果标题中包含"理论"、"方法"、"治疗"等关键词，请在相关句子中明确指出这是一种理论、方法或治疗手段
            6. 对于医学专业术语，保留其完整表述，不要简化或使用代词替代
            7. 输出格式必须是包含sentences字段的JSON对象

            输入示列："1678年，医学教授乔治·弗兰克·冯·弗兰克瑙（Georg Franck von Franckenau）在德国西南部记录了复活节野兔（Osterhase）的最早证据，但直到18世纪，德国其他地区都知道它的存在。"

            输出示例: {{
                "sentences": [
                    "1678年，医学教授Georg Franck von Franckenau在德国西南部首次记录了复活节兔子的最早证据。",
                    "Georg Franck von Franckenau是一位医学教授。",
                    "复活节兔子理论直到18世纪才在德国其他地区被人们知晓。",
                    "复活节兔子理论与解释是由Georg Franck von Franckenau首次提出的。"
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


def run_docs_chunk(essay, gid = None):

    """
    essay: 文档内容(Json格式)
    """
    # obj = hub.pull("wfh/proposal-indexing")


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
            3. 完全消除指代词，不要使用"它"、"他"、"她"、"他们"、"这个"、"那个"等代词，而是重复使用具体的名词
            4. 使用标题中的主题词作为句子的主语，确保每个句子都有明确的主语
            5. 如果标题中包含"理论"、"方法"、"治疗"、"管理"、"诊断"等关键词，请在相关句子中明确指出这是一种理论、方法、治疗手段、管理策略或诊断方法
            6. 对于医学专业术语，保留其完整表述，不要简化或使用代词替代
            7. 对于医学指南中的建议或推荐，明确指出这是指南的建议，而非一般性陈述
            8. 输出格式必须是包含sentences字段的JSON对象

            医学示例输入："Title：二、糖尿病肾病的诊断与治疗。content：早期糖尿病肾病通常无症状，它的诊断主要依靠尿微量白蛋白和肾小球滤过率的检测。当它进展到临床期，患者可能出现水肿、高血压和肾功能下降。治疗上，应控制血糖、血压，并使用ACEI或ARB类药物减少蛋白尿。"

            医学示例输出: {{
                "sentences": [
                    "早期糖尿病肾病通常无症状。",
                    "糖尿病肾病的诊断主要依靠尿微量白蛋白和肾小球滤过率的检测。",
                    "当糖尿病肾病进展到临床期时，患者可能出现水肿。",
                    "当糖尿病肾病进展到临床期时，患者可能出现高血压。",
                    "当糖尿病肾病进展到临床期时，患者可能出现肾功能下降。",
                    "糖尿病肾病的治疗方法包括控制血糖。",
                    "糖尿病肾病的治疗方法包括控制血压。",
                    "糖尿病肾病的治疗方法包括使用ACEI类药物减少蛋白尿。",
                    "糖尿病肾病的治疗方法包括使用ARB类药物减少蛋白尿。"
                ]
            }}

            输入示列："Title：一、理论与解释，与复活节兔子的关系。content：1678年，医学教授乔治·弗兰克·冯·弗兰克瑙（Georg Franck von Franckenau）在德国西南部记录了复活节野兔（Osterhase）的最早证据，但直到18世纪，德国其他地区都知道它的存在。"

            输出示例: {{
                "sentences": [
                    "1678年，医学教授Georg Franck von Franckenau在德国西南部首次记录了复活节兔子的最早证据。",
                    "Georg Franck von Franckenau是一位医学教授。",
                    "复活节兔子理论直到18世纪才在德国其他地区被人们知晓。",
                    "复活节兔子理论与解释是由Georg Franck von Franckenau首次提出的。"
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

    chunks = {}
    """
    每段提取关键句子，使用多线程并行处理
    """

    
    # 创建线程锁，用于保护chunks字典的并发访问
    chunks_lock = threading.Lock()
    
    def process_item(item_tuple):
        i, item = item_tuple
        para = item['content']
        title = item['section_title']
        # content = title:title + content:content
        content = 'title:' + title + ' content:' + para
        propositions = get_propositions(content, runnable, extraction_chain)
        # 新增
        item_copy = item.copy()  # 创建副本避免修改原始数据
        item_copy['propositions'] = propositions
        
        # 使用线程锁保护共享资源
        with chunks_lock:
            new_chunk = _create_new_chunk_mt(chunks, item_copy, len(chunks))
            logger.info(f"Done with {i}")
        return new_chunk
    
    # 定义一个新的创建chunk函数，返回创建的chunk而不是直接修改全局变量
    def _create_new_chunk_mt(chunks_dict, item, index):
        new_chunk_id = str(uuid.uuid4())[:5]  # 生成短UUID
        new_chunk = {
            'chunk_id': new_chunk_id,
            'content': item.get('content'),
            'propositions': item.get('propositions'),
            'section_title': item.get('section_title'),
            'chunk_index': index
        }
        chunks_dict[new_chunk_id] = new_chunk
        logger.info(f"Created new chunk with ID: {new_chunk_id}")
        return new_chunk
    
    # 确定合适的线程数量，通常为CPU核心数的2-4倍
    max_workers = min(32, (os.cpu_count() or 4) * 2)
    
    # 添加日志记录
    logger.info(f"Starting parallel processing with {max_workers} workers for {len(essay)} items")
    
    # 添加异常处理的包装函数
    def process_item_safe(item_tuple):
        try:
            return process_item(item_tuple)
        except Exception as e:
            logger.error(f"Error processing item {item_tuple[0]}: {str(e)}")
            # 返回一个带有错误信息的占位结果
            i, item = item_tuple
            item_copy = item.copy()
            item_copy['propositions'] = [f"处理错误: {str(e)}"]
            return item_copy
    
    # 检查essay是否为空
    if not essay:
        logger.warning("Essay is empty, no items to process")
        return chunks
    
    # 使用ThreadPoolExecutor并行处理所有项目，添加异常处理
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 将索引和项目打包在一起传递给处理函数，使用安全的处理函数
            results = list(executor.map(process_item_safe, enumerate(essay)))
            logger.info(f"Successfully processed {len(results)} items")
    except Exception as e:
        logger.error(f"Error in thread pool execution: {str(e)}")
        # 如果线程池执行失败，回退到顺序处理
        logger.info("Falling back to sequential processing")
        results = []
        for item_tuple in enumerate(essay):
            try:
                result = process_item(item_tuple)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in sequential processing for item {item_tuple[0]}: {str(e)}")
    
    logger.info(f"Processed {len(results)} items with {len(chunks)} chunks")
    return chunks

# 消歧　
def eliminate_ambiguity_chunk(essay, gid = None):

    """
    essay: 文档内容(Json格式)
    """
    # obj = hub.pull("wfh/proposal-indexing")


    llm = init_chat_model(            
        model="deepseek-ai/DeepSeek-V2.5", 
        model_provider="openai",
        api_key="sk-wzahzejbmyrpdluvpopvijotobnslviwgnbildslbekjausu",
        base_url="https://api.siliconflow.cn/v1"
    )
    obj = ChatPromptTemplate.from_messages(
        [
            ("system", """
            将以下input分解为清晰简单的陈述句，确保这些句子在脱离上下文的情况下也能被理解。

            指南:
            1. 将复合句分解为简单句，尽可能保持原文的表述方式
            2. 对于包含附加描述信息的命名实体，将这些信息分离为独立的陈述句
            3. 完全消除指代词，不要使用"它"、"他"、"她"、"他们"、"这个"、"那个"等代词，而是重复使用具体的名词
            4. 使用标题中的主题词作为句子的主语，确保每个句子都有明确的主语
            5. 如果标题中包含"理论"、"方法"、"治疗"、"管理"、"诊断"等关键词，请在相关句子中明确指出这是一种理论、方法、治疗手段、管理策略或诊断方法
            6. 对于医学专业术语，保留其完整表述，不要简化或使用代词替代
            7. 对于医学指南中的建议或推荐，明确指出这是指南的建议，而非一般性陈述
            8. 输出格式必须是包含sentences字段的JSON对象

            医学示例输入："Title：二、糖尿病肾病的诊断与治疗。content：早期糖尿病肾病通常无症状，它的诊断主要依靠尿微量白蛋白和肾小球滤过率的检测。当它进展到临床期，患者可能出现水肿、高血压和肾功能下降。治疗上，应控制血糖、血压，并使用ACEI或ARB类药物减少蛋白尿。"

            医学示例输出: {{
                "sentences": [
                    "早期糖尿病肾病通常无症状。",
                    "糖尿病肾病的诊断主要依靠尿微量白蛋白和肾小球滤过率的检测。",
                    "当糖尿病肾病进展到临床期时，患者可能出现水肿。",
                    "当糖尿病肾病进展到临床期时，患者可能出现高血压。",
                    "当糖尿病肾病进展到临床期时，患者可能出现肾功能下降。",
                    "糖尿病肾病的治疗方法包括控制血糖。",
                    "糖尿病肾病的治疗方法包括控制血压。",
                    "糖尿病肾病的治疗方法包括使用ACEI类药物减少蛋白尿。",
                    "糖尿病肾病的治疗方法包括使用ARB类药物减少蛋白尿。"
                ]
            }}

            输入示列："Title：一、理论与解释，与复活节兔子的关系。content：1678年，医学教授乔治·弗兰克·冯·弗兰克瑙（Georg Franck von Franckenau）在德国西南部记录了复活节野兔（Osterhase）的最早证据，但直到18世纪，德国其他地区都知道它的存在。"

            输出示例: {{
                "sentences": [
                    "1678年，医学教授Georg Franck von Franckenau在德国西南部首次记录了复活节兔子的最早证据。",
                    "Georg Franck von Franckenau是一位医学教授。",
                    "复活节兔子理论直到18世纪才在德国其他地区被人们知晓。",
                    "复活节兔子理论与解释是由Georg Franck von Franckenau首次提出的。"
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

    extraction_chain = llm.with_structured_output(Sentences)
    
    essay_propositions = []
    """
    每段提取关键句子，使用多线程并行处理
    """
    propositions = get_propositions(essay, runnable, extraction_chain)
    essay_propositions.extend(propositions)

    return essay_propositions

def split_sentences(text):
    """
    自动将中文文本分句，返回句子列表
    """
    # 使用正则表达式按句号、问号、感叹号、分号、换行等断句
    sentences = re.split(r'[。！？；\n]+', text)
    # 去除空白和空字符串
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

if __name__ == "__main__":
    text = "1.女患，51岁。 2.主  诉: 血糖升高7年，控制不佳4天。3.既往史: 否认高血压及心脏病史，否认乙肝结核病史，否认药物过敏史，可疑家族遗传糖尿病史。 4.体  检：体温：36.2℃,脉搏:78次/分,呼吸:20次/分,血压:110/80mmHg，身高155cm，体重64kg,BMI 26.6kg/㎡,神志清楚，眼睛胀痛，右眼视物模糊。右足背动脉搏动减弱，左侧正常。 5.门诊资料: 入院心电图示：窦性心律，正常心电图；随机血糖12.0mmol/L；2019.07.29孝感市中心医院颈椎X线示：颈椎骨质增生；小关节退行性变。尿常规:蛋白质 +- 、葡萄糖 2+ 。2019-07-31)血糖测定:葡萄糖 12.12 mmol/L↑。血脂:低密度脂蛋白 3.83 mmol/L↑、脂蛋白a 445 mg/L↑。肝功能:谷丙转氨酶 37 U/L↑。甲状腺功能(三项）:未见明显异常。2019.07.31双下肢动脉彩超示：双下肢动脉可显示段未见明显异常；双下肢深静脉可显示段血流通畅。颈部血管超声示：双侧椎动脉管径发育不对称，右侧发育偏细。心脏超声示：左室舒张功能减低。糖化血红蛋白 9% 5.诊治经过：1.完善相关检查（三大常规、肝肾功电解质、血脂、糖化血红蛋白）；2.降糖，扩管，对症支持治疗；3.根据病情变化调整治疗方案。 入院后予以降糖，扩管，对症支持治疗，患者病情好转，予以办理出院。6.诊断结果：2型糖尿病（并糖尿病肾病并视网膜病变）"
    sentences = split_sentences(text)
    print(sentences)


