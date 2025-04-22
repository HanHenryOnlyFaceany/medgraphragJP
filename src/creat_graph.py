import os
import json
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
from camel.configs import PPIOConfig, OllamaConfig, ChatGPTConfig


def get_checkpoint_path(gid):
    """获取检查点文件路径
    Args:
        gid: 图元素组ID
    Returns:
        str: 检查点文件路径
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"checkpoint_{gid}.json")

def save_checkpoint(gid, current_index, total_chunks, processed_chunks=None):
    """保存处理进度到检查点文件
    Args:
        gid: 图元素组ID
        current_index: 当前处理的块索引
        total_chunks: 总块数
        processed_chunks: 已处理的块索引列表
    """
    checkpoint_path = get_checkpoint_path(gid)
    checkpoint_data = {
        "gid": gid,
        "current_index": current_index,
        "total_chunks": total_chunks,
        "processed_chunks": processed_chunks if processed_chunks else [],
        "timestamp": time.time()
    }
    
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f)
    print(f"检查点已保存: {checkpoint_path}")

def load_checkpoint(gid):
    """加载检查点文件
    Args:
        gid: 图元素组ID
    Returns:
        dict: 检查点数据，如果不存在则返回None
    """
    checkpoint_path = get_checkpoint_path(gid)
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
            print(f"已加载检查点: {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            print(f"加载检查点失败: {e}")
    return None


def grained_chunk(content, gid, checkpoint=None, chunks_cache_path=None):
    # 如果存在检查点，恢复处理进度
    if checkpoint and checkpoint["gid"] == gid:
        processed_chunks = set(checkpoint["processed_chunks"])
        if processed_chunks:
            #根据先前保存的文件，导出json文件为list，作为content_chunks
            with open(chunks_cache_path, "r", encoding='utf-8') as f:
                loaded_chunks = json.load(f)
                chunks = []
                for chunk in loaded_chunks.values():
                    # 将 propositions 列表连接成单个字符串
                    chunks.append(" ".join(chunk['propositions']))
                content_chunks = chunks
            print(f"从检查点恢复，已处理 {len(processed_chunks)}/{len(content_chunks)} 个块")

    else:
        print("执行细粒度分块...")
        content_chunks = run_chunk(content, gid)
    return content_chunks




def process_chunks_with_checkpoint(content_chunks, processed_chunks, gid, uio, kg_agent, n4j):
    """
    处理每个内容块，并在每步后保存检查点
    Args:
        content_chunks: 内容块列表
        processed_chunks: 已处理块索引集合
        gid: 图元素组ID
        uio: UnstructuredIO实例
        kg_agent: KnowledgeGraphAgent实例
        n4j: Neo4jGraph实例
    """
    for i, cont in enumerate(content_chunks, 1):
        # 如果该块已处理，则跳过
        if i in processed_chunks:
            print(f"跳过已处理的块 {i}/{len(content_chunks)}")
            continue

        try:
            print(f"\n处理第 {i}/{len(content_chunks)} 个内容块...")
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

            # 更新并保存检查点
            processed_chunks.add(i)
            save_checkpoint(gid, i, len(content_chunks), list(processed_chunks))
        except Exception as e:
            print(f"处理块 {i} 时出错: {e}")
            # 保存当前进度，以便稍后恢复
            save_checkpoint(gid, i, len(content_chunks), list(processed_chunks))
            raise

def creat_metagraph(args, content, gid, n4j):
    print(f"\n=== 开始处理新的图元素组 (GID: {gid}) ===")
    
    os.environ["PPIO_API_KEY"] = "sk_J7l6-K7MQB_9aPomZCuXWXrmxEUF_U91rXvGfRypmj0"
    os.environ["PPIO_API_BASE"] = "https://api.ppinfra.com/v3/openai"

    # Set instance
    print("正在初始化 PPIO 模型...")
    PPIO_DP_V_3_turbo = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=os.environ.get("OLLAMA_MODEL"),
        api_key=os.environ.get("OLLAMA_API_KEY"),
        url="http://ai.medical-deep.com:20240/v1",
        model_config_dict=ChatGPTConfig(temperature=0.2).as_dict(),
    )

    print("PPIO 模型初始化完成")

    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent(model=PPIO_DP_V_3_turbo)
    whole_chunk = content
    
    # 检查是否存在检查点
    checkpoint = load_checkpoint(gid)
    processed_chunks = set()
    chunks_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", f"chunks_{gid}.json")
    if checkpoint and checkpoint["gid"] == gid:
        start_index = checkpoint["current_index"]
        processed_chunks = set(checkpoint["processed_chunks"])


    if args.grained_chunk == True:
        content_chunks = grained_chunk(content, gid, checkpoint, chunks_cache_path)
        print(f"分块完成，共生成 {len(content_chunks)} 个块")
    else:
        content_chunks = [content]
        print("使用原始内容，不进行分块")

    # 调用新方法处理内容块
    process_chunks_with_checkpoint(content_chunks, processed_chunks, gid, uio, kg_agent, n4j)

    # 检查是否所有块都已处理
    if len(processed_chunks) == len(content_chunks):
        if args.ingraphmerge:
            print("\n开始合并相似节点...")
            merge_similar_nodes(n4j, gid)
            print("节点合并完成")

        print("\n生成内容摘要...")
        add_sum(n4j, whole_chunk, gid)
        print("摘要生成完成")
        
        # 处理完成后删除检查点文件
        checkpoint_path = get_checkpoint_path(gid)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"检查点已删除: {checkpoint_path}")
        
        #可以选择保留或删除分块缓存
        if os.path.exists(chunks_cache_path):
            os.remove(chunks_cache_path)
            print(f"分块缓存已删除: {chunks_cache_path}")
    
    print(f"\n=== 图元素组 (GID: {gid}) 处理完成 ===\n")
    return n4j

