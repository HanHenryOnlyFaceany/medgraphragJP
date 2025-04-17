from openai import OpenAI
import os
from neo4j import GraphDatabase
import numpy as np
from camel.storages import Neo4jGraph
import uuid
from summerize import process_chunks
import openai

"""
系统提示词，用于生成基于图数据的医疗信息回答
"""
sys_prompt_one = """
Please answer the question using insights supported by provided graph-based data relevant to medical information.
"""

"""
系统提示词，用于根据参考信息修改和完善回答
"""
sys_prompt_two = """
Modify the response to the question using the provided references. Include precise citations relevant to your answer. You may use multiple citations simultaneously, denoting each with the reference index number. For example, cite the first and third documents as [1][3]. If the references do not pertain to the response, simply provide a concise answer to the original question.
"""

# Add your own OpenAI API key
openai_api_key = os.getenv("PPIO_API_KEY")
openai_api_url = os.getenv("PPIO_API_BASE")


def get_embedding(text, mod = "baai/bge-m3"):
    client = OpenAI(
        api_key = "sk_J7l6-K7MQB_9aPomZCuXWXrmxEUF_U91rXvGfRypmj0",
        base_url = "https://api.ppinfra.com/v3/openai",
    )

    response = client.embeddings.create(
        input=text,
        model=mod
    )

    return response.data[0].embedding

def fetch_texts(n4j):
    """
    从Neo4j数据库中获取所有节点的文本内容
    Args:
        n4j: Neo4j数据库连接对象
    Returns:
        list: 包含所有节点ID的列表
    """
    query = "MATCH (n) RETURN n.id AS id"
    return n4j.query(query)

def add_embeddings(n4j, node_id, embedding):
    """
    将嵌入向量添加到Neo4j节点中
    Args:
        n4j: Neo4j数据库连接对象
        node_id: 节点ID
        embedding: 嵌入向量
    """
    query = "MATCH (n) WHERE n.id = $node_id SET n.embedding = $embedding"
    n4j.query(query, params = {"node_id":node_id, "embedding":embedding})

def add_nodes_emb(n4j):
    """
    为数据库中的所有节点添加嵌入向量
    Args:
        n4j: Neo4j数据库连接对象
    """
    nodes = fetch_texts(n4j)
    for node in nodes:
        if node['id']:
            embedding = get_embedding(node['id'])
            add_embeddings(n4j, node['id'], embedding)

def add_ge_emb(graph_element):
    """
    为图元素中的所有节点添加嵌入向量
    Args:
        graph_element: 图元素对象
    Returns:
        object: 更新后的图元素对象
    """
    for node in graph_element.nodes:
        emb = get_embedding(node.id)
        node.properties['embedding'] = emb
    return graph_element

def add_gid(graph_element, gid):
    """
    为图元素中的所有节点和关系添加组ID
    Args:
        graph_element: 图元素对象
        gid: 组ID
    Returns:
        object: 更新后的图元素对象
    """
    for node in graph_element.nodes:
        node.properties['gid'] = gid
    for rel in graph_element.relationships:
        rel.properties['gid'] = gid
    return graph_element

def add_sum(n4j, content, gid):
    """
    为内容创建摘要节点并建立关系
    Args:
        n4j: Neo4j数据库连接对象
        content: 需要总结的内容
        gid: 组ID
    Returns:
        object: 创建的摘要节点
    """
    sum = process_chunks(content)
    creat_sum_query = """
        CREATE (s:Summary {content: $sum, gid: $gid})
        RETURN s
        """
    s = n4j.query(creat_sum_query, {'sum': sum, 'gid': gid})
    
    link_sum_query = """
        MATCH (s:Summary {gid: $gid}), (n)
        WHERE n.gid = s.gid AND NOT n:Summary
        CREATE (s)-[:SUMMARIZES]->(n)
        RETURN s, n
        """
    n4j.query(link_sum_query, {'gid': gid})
    return s

def call_llm(sys, user, model_config=None):
    """
    调用大语言模型生成回答，支持多种模型提供商
    Args:
        sys: 系统提示词
        user: 用户输入
        model_config: 模型配置字典，包含provider、model等信息
    Returns:
        str: 模型生成的回答
    """
    # 默认配置
    config = {
        "provider": "openai",
        "model": "qwen/qwen-2.5-72b-instruct", 
        "max_tokens": 500,
        "temperature": 0.5,
        "api_key": "sk_J7l6-K7MQB_9aPomZCuXWXrmxEUF_U91rXvGfRypmj0",
        "base_url": "https://api.ppinfra.com/v3/openai"
    }
    
    # 更新配置
    if model_config:
        config.update(model_config)
    
    # 构建消息
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f" {user}"},
    ]
    
    # 根据提供商选择不同的实现
    if config["provider"] == "openai":
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        response = client.chat.completions.create(
            model=config["model"],
            messages=messages,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        return response.choices[0].message.content
    
    # 可以在这里添加其他提供商的实现
    # elif config["provider"] == "其他提供商":
    #     ...
    
    else:
        raise ValueError(f"不支持的提供商: {config['provider']}")

def find_index_of_largest(nums):
    """
    找出列表中最大值的索引
    Args:
        nums: 数字列表
    Returns:
        int: 最大值的索引
    """
    sorted_with_index = sorted((num, index) for index, num in enumerate(nums))
    largest_original_index = sorted_with_index[-1][1]
    return largest_original_index

def get_response(n4j, gid, query):
    """
    根据查询生成综合回答
    Args:
        n4j: Neo4j数据库连接对象
        gid: 组ID
        query: 查询问题
        content: 可选的医疗索引内容
    Returns:
        str: 生成的回答
    """

    selfcont = ret_context(n4j, gid)
    selfsum = selfsum_context(n4j, gid)
    linkcont = link_context(n4j, gid)
    user_one = "the question is: " + query + "the provided information is:" +  "".join(selfcont) + "the medical indexs are: " + selfsum
    res = call_llm(sys_prompt_one, user_one)
    user_two = "the question is: " + query + "the last response of it is:" + res + "the references are: " + "".join(linkcont)
    res = call_llm(sys_prompt_two, user_two)
    return res

def link_context(n4j, gid):
    """
    获取跨图的引用关系上下文
    Args:
        n4j: Neo4j数据库连接对象
        gid: 组ID
    Returns:
        list: 引用关系的描述列表
    """
    cont = []
    retrieve_query = """
        // Match all 'n' nodes with a specific gid but not of the "Summary" type
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary

        // Find all 'm' nodes where 'm' is a reference of 'n' via a 'REFERENCES' relationship
        MATCH (n)-[r:REFERENCE]->(m)
        WHERE NOT m:Summary

        // Find all 'o' nodes connected to each 'm', and include the relationship type,
        // while excluding 'Summary' type nodes and 'REFERENCE' relationship
        MATCH (m)-[s]-(o)
        WHERE NOT o:Summary AND TYPE(s) <> 'REFERENCE'

        // Collect and return details in a structured format
        RETURN n.id AS NodeId1, 
            m.id AS Mid, 
            TYPE(r) AS ReferenceType, 
            collect(DISTINCT {RelationType: type(s), Oid: o.id}) AS Connections
    """
    res = n4j.query(retrieve_query, {'gid': gid})
    for r in res:
        for ind, connection in enumerate(r["Connections"]):
            cont.append("Reference " + str(ind) + ": " + r["NodeId1"] + "has the reference that" + r['Mid'] + connection['RelationType'] + connection['Oid'])
    return cont

def ret_context(n4j, gid):
    """
    获取同组内节点间的关系上下文
    Args:
        n4j: Neo4j数据库连接对象
        gid: 组ID
    Returns:
        list: 节点关系的描述列表
    """
    cont = []
    ret_query = """
    // Match all nodes with a specific gid but not of type "Summary" and collect them
    MATCH (n)
    WHERE n.gid = $gid AND NOT n:Summary
    WITH collect(n) AS nodes

    // Unwind the nodes to a pairs and match relationships between them
    UNWIND nodes AS n
    UNWIND nodes AS m
    MATCH (n)-[r]-(m)
    WHERE n.gid = m.gid AND id(n) < id(m) AND NOT n:Summary AND NOT m:Summary
    WITH n, m, TYPE(r) AS relType

    // Return node IDs and relationship types in structured format
    RETURN n.id AS NodeId1, relType, m.id AS NodeId2
    """
    res = n4j.query(ret_query, {'gid': gid})
    for r in res:
        cont.append(r['NodeId1'] + r['relType'] + r['NodeId2'])
    return cont

def selfsum_context(n4j, gid):
    """
    获取医学指标结构上下文
    Args:
        n4j: Neo4j数据库连接对象
        gid: 组ID
    Returns:
        list: 节点关系的描述列表
    """
    cont = []
    query = """
    MATCH (s:Summary)
    WHERE s.gid = $gid
    RETURN s.content as content
    """
    content = n4j.query(query, {"gid": gid})
        # 如果提供了content，将其转换为字符串
    if content:
        if isinstance(content, dict):
            cont = str(content)
        elif isinstance(content, list):
            cont = "".join(str(item) for item in content)
        else:
            cont = str(content)
    return cont

def merge_similar_nodes(n4j, gid):
    """
    合并相似的节点
    Args:
        n4j: Neo4j数据库连接对象
        gid: 组ID（可选）
    Returns:
        object: 合并操作的结果
    """
    if gid:
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary AND n.gid = m.gid AND n.gid = $gid AND n<>m AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
            WITH n, m,
                gds.similarity.cosine(n.embedding, m.embedding) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*)
        """
        result = n4j.query(merge_query, {'gid': gid})
    else:
        merge_query = """
            WITH 0.5 AS threshold
            MATCH (n), (m)
            WHERE NOT n:Summary AND NOT m:Summary AND n<>m AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
            WITH n, m,
                gds.similarity.cosine(n.embedding, m.embedding) AS similarity
            WHERE similarity > threshold
            WITH head(collect([n,m])) as nodes
            CALL apoc.refactor.mergeNodes(nodes, {properties: 'overwrite', mergeRels: true})
            YIELD node
            RETURN count(*)
        """
        result = n4j.query(merge_query)
    return result

def ref_link(n4j, gid1, gid2):
    """
    建立两个图之间的引用关系(余弦相似度)
    Args:
        n4j: Neo4j数据库连接对象
        gid1: 第一个图的组ID
        gid2: 第二个图的组ID
    Returns:
        object: 创建的引用关系
    """
    trinity_query = """
        // Match nodes from Graph A
        MATCH (a)
        WHERE a.gid = $gid1 AND NOT a:Summary
        WITH collect(a) AS GraphA

        // Match nodes from Graph B
        MATCH (b)
        WHERE b.gid = $gid2 AND NOT b:Summary
        WITH GraphA, collect(b) AS GraphB

        // Unwind the nodes to compare each against each
        UNWIND GraphA AS n
        UNWIND GraphB AS m

        // Set the threshold for cosine similarity
        WITH n, m, 0.6 AS threshold

        // Compute cosine similarity and apply the threshold
        WHERE apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m)) AND n <> m
        WITH n, m, threshold,
            gds.similarity.cosine(n.embedding, m.embedding) AS similarity
        WHERE similarity > threshold

        // Create a relationship based on the condition
        MERGE (m)-[:REFERENCE]->(n)

        // Return results
        RETURN n, m
    """
    result = n4j.query(trinity_query, {'gid1': gid1, 'gid2': gid2})
    return result

def str_uuid():
    """
    生成UUID字符串
    Returns:
        str: 生成的UUID字符串
    """
    generated_uuid = uuid.uuid4()
    return str(generated_uuid)


