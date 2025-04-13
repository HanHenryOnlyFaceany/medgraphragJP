import os
from getpass import getpass
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from dataloader import load_high
import argparse
from data_chunk import run_chunk
from utils import *

import time
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import PPIOConfig, OllamaConfig


def creat_metagraph(args, content, gid, n4j):
    print(f"\n=== 开始处理新的图元素组 (GID: {gid}) ===")
    
    os.environ["PPIO_API_KEY"] = "sk_J7l6-K7MQB_9aPomZCuXWXrmxEUF_U91rXvGfRypmj0"
    os.environ["PPIO_API_BASE"] = "https://api.ppinfra.com/v3/openai"
    
    # Set instance
    print("正在初始化 PPIO 模型...")
    PPIO_DP_V_3_turbo = ModelFactory.create(
        model_platform=ModelPlatformType.PPIO,
        model_type=ModelType.PPIO_QWEN_2_5_72B,
        model_config_dict=PPIOConfig(temperature=0.2).as_dict(),
    )
    print("PPIO 模型初始化完成")

    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent(model=PPIO_DP_V_3_turbo)
    whole_chunk = content

    if args.grained_chunk == True:
        print("执行细粒度分块...")
        content = run_chunk(content)
        print(f"分块完成，共生成 {len(content)} 个块")
    else:
        content = [content]
        print("使用原始内容，不进行分块")

    for i, cont in enumerate(content, 1):
        print(f"\n处理第 {i}/{len(content)} 个内容块...")
        element_example = uio.create_element_from_text(text=cont)
        
        print("生成知识图谱元素...")
        graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
        print("添加嵌入向量...")
        graph_elements = add_ge_emb(graph_elements)
        print("添加组ID...")
        graph_elements = add_gid(graph_elements, gid)
        
        print("将图元素添加到数据库...")
        n4j.add_graph_elements(graph_elements=[graph_elements])
        print(f"第 {i} 个内容块处理完成")

    if args.ingraphmerge:
        print("\n开始合并相似节点...")
        merge_similar_nodes(n4j, gid)
        print("节点合并完成")

    print("\n生成内容摘要...")
    add_sum(n4j, whole_chunk, gid)
    print("摘要生成完成")
    
    print(f"\n=== 图元素组 (GID: {gid}) 处理完成 ===\n")
    return n4j

