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
    os.environ["PPIO_API_KEY"] = "sk_J7l6-K7MQB_9aPomZCuXWXrmxEUF_U91rXvGfRypmj0"
    os.environ["PPIO_API_BASE"] = "https://api.ppinfra.com/v3/openai"

    
    # Set instance
    PPIO_DP_V_3_turbo = ModelFactory.create(
        model_platform=ModelPlatformType.PPIO,
        model_type=ModelType.PPIO_QWEN_2_5_72B,
        model_config_dict=PPIOConfig(temperature=0.2).as_dict(),
    )


    uio = UnstructuredIO()
    kg_agent = KnowledgeGraphAgent(model=PPIO_DP_V_3_turbo)
    whole_chunk = content

    if args.grained_chunk == True:
        content = run_chunk(content)
    else:
        content = [content]
    for cont in content:
        element_example = uio.create_element_from_text(text=cont)

        # ans_str = kg_agent.run(element_example, parse_graph_elements=False)
        # print(ans_str)

        graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
        graph_elements = add_ge_emb(graph_elements)
        graph_elements = add_gid(graph_elements, gid)
        print(graph_elements)

        n4j.add_graph_elements(graph_elements=[graph_elements])
    if args.ingraphmerge:
        merge_similar_nodes(n4j, gid)
    add_sum(n4j, whole_chunk, gid)
    return n4j

