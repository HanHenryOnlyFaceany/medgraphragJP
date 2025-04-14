import os
from getpass import getpass
from camel.storages import Neo4jGraph
from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from dataloader import load_high
import argparse
from data_chunk import run_chunk
from creat_graph import creat_metagraph
from summerize import process_chunks
from retrieve import seq_ret
from utils import *
# from nano_graphrag import GraphRAG, QueryParam


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-simple', action='store_true')
parser.add_argument('-construct_graph', action='store_true')
parser.add_argument('-inference',  action='store_true')
parser.add_argument('-grained_chunk',  action='store_true')
parser.add_argument('-trinity', action='store_true')
parser.add_argument('-trinity_gid1', type=str)
parser.add_argument('-trinity_gid2', type=str)
parser.add_argument('-ingraphmerge',  action='store_true')
parser.add_argument('-crossgraphmerge', action='store_true')
parser.add_argument('-dataset', type=str, default='mimic_ex')
parser.add_argument('-data_path', type=str, default='./dataset_cn')
parser.add_argument('-test_data_path', type=str, default='./dataset_cn/report_0.txt')
parser.add_argument('query', nargs='?', type=str, help="Query to run in simple mode")
args = parser.parse_args()

if args.simple:
    fixed_text = ""
    graph_func = GraphRAG(working_dir="./nanotest")

    with open("./dataset_ex/report_0.txt") as f:
        graph_func.insert(f.read())

        if args.query:
            full_query = f"{fixed_text}{args.query}"
            # Run the query from the command-line argument
            print(graph_func.query(full_query, param=QueryParam(mode="local")))
        else:
            print("Please provide a query when using -simple mode.")

        # # Perform local graphrag search (I think is better and more scalable one)
        # print(graph_func.query("don't add extra information in the response. give me the list of medication he takes?", param=QueryParam(mode="local")))

else:

    url=os.getenv("NEO4J_URL")
    username=os.getenv("NEO4J_USERNAME")
    password=os.getenv("NEO4J_PASSWORD")
    n4j = Neo4jGraph(
        url=url,
        username=username,             # Default username
        password=password     # Replace 'yourpassword' with your actual password
    )
    # 
    if args.construct_graph: 
        if args.dataset == 'mimic_ex':
            files = [file for file in os.listdir(args.data_path) if os.path.isfile(os.path.join(args.data_path, file))]
            
            # Read and print the contents of each file
            for file_name in files:
                file_path = os.path.join(args.data_path, file_name)
                content = load_high(file_path)
                gid = str_uuid()
                
                n4j = creat_metagraph(args, content, gid, n4j)

                """
                初始逻辑：
                    对应的连接关系需要我们在图构建的时候就定好，多个gid即trinity_gid1/2/3/4
                    所以我们在构建的时候就需要做好底层的连接关系
                改进逻辑：
                    在这个地方可以加一个结构化存储，再通过mcp的形式，匹配对应的关系
                """
                if args.trinity:
                    # link_context(n4j, args.trinity_gid1)
                    ref_link(n4j,gid,args.trinity_gid1)
            if args.crossgraphmerge:
                merge_similar_nodes(n4j, None)

    if args.inference:
        question = load_high("./prompt.txt")
        sum = process_chunks(question)
        gid = seq_ret(n4j, sum)
        response = get_response(n4j, gid, question)
        print(response)
