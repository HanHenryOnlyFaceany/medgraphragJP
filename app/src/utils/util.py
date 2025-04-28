from textwrap import indent
from openai import OpenAI
import os
from neo4j import GraphDatabase
import numpy as np
from camel.storages import Neo4jGraph
import uuid
from src.summerize import process_chunks
import json
from src.construct import *

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

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

"""
系统提示词，用于生成基于reference的图数据的医疗信息回答并用于根据参考信息修改和完善回答
"""
quick_sys_prompt_one = """
请基于提供的医学图谱数据回答问题，并利用引用信息增强回答的准确性。在回答中，请精确引用相关文献，可同时使用多个引用，使用引用编号标注（例如：[1][3]表示引用第一篇和第三篇文献,对文献进行去重，即多个三元组来自同一个文献及引用一个文献）。如果提供的引用与问题无关，请直接提供简洁的回答。确保回答专业、准确且易于理解。
"""

"""
系统提示词，用于根据医学字典的参考信息修改和完善回答
"""
quick_sys_prompt_two = """
请根据提供的医学字典参考信息，修改和完善您的回答。确保使用标准医学术语，并在必要时引用医学字典中的定义。如果医学字典中的信息与您之前的回答有冲突，请以医学字典为准。保持回答的专业性和准确性，同时确保内容对非专业人士也易于理解。
"""

# Add your own OpenAI API key
openai_api_key = os.getenv("PPIO_API_KEY")
openai_api_url = os.getenv("PPIO_API_BASE")


def get_embedding(text, mod = "m3e-base"):
    client = OpenAI(
        api_key = "sk-cJBVEQFNNxphN4Et84F5A9C0083348C6873443A27a18C3De",
        base_url = "http://ai.medical-deep.com:20240/v1",
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

def add_embeddings_tr(session, node_name, embedding):
    """
    将嵌入向量添加到Neo4j节点中
    Args:
        session: Neo4j会话对象
        node_name: 节点名称
        embedding: 嵌入向量
    """
    query = "MATCH (n) WHERE n.name = $node_name SET n.embedding = $embedding"
    session.run(query, node_name=node_name, embedding=embedding)

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

def construct_kg(construct, extraction_result, gid=None, chunk_id=None):
    myurl = construct['url']
    myusername = construct['username']
    mypassword = construct['password']
    try:
        print(f"Construct KG in your {construct['database']} now...")
        cypher_statements = generate_cypher_statements(extraction_result, gid, chunk_id)
        execute_cypher_statements(uri=myurl, user=myusername, password=mypassword, cypher_statements=cypher_statements)
    except Exception as e:
        print(f"Error constructing KG: {e}")

def add_nodes_emb_tr(session, gid):
    """
    为数据库中的所有节点添加嵌入向量
    Args:
        session: Neo4j会话对象
    """
    query = "MATCH (n {gid: $gid}) RETURN n.name AS name"
    result = session.run(query, {'gid': gid})
    nodes = [record for record in result]

    for node in nodes:
        if node.get('name'):
            embedding = get_embedding(node['name'])
            if embedding:
                add_embeddings_tr(session, node['name'], embedding)

def add_all_embeddings(construct, gid):
    myurl = construct['url']
    myusername = construct['username']
    mypassword = construct['password']
    driver = GraphDatabase.driver(
        myurl, auth=(myusername, mypassword)
    )
    with driver.session() as session:
        add_nodes_emb_tr(session, gid)
    driver.close()

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

    
def add_chunk(n4j, gid, chunk_id, content):

    creat_sum_query = """
        CREATE (s:chunk {chunk_id: $chunk_id, content: $content, gid: $gid})
        RETURN s
        """
    s = n4j.query(creat_sum_query, {'chunk_id': chunk_id, 'content': content, 'gid': gid})
    return s

def add_section(n4j, gid, chunk_id, section):
    """
    为所有属性gid=gid，chunk_id=chunk_id的节点添加section属性
    Args:
        n4j: Neo4j数据库连接对象
        gid: 组ID
        section: 章节内容
        chunk_id: chunk的ID
    Returns:
        object: 更新后的节点
    """
    creat_sum_query = """
        MATCH (s:chunk {gid: $gid, chunk_id: $chunk_id})
        SET s.section = $section
        RETURN s
        """
    s = n4j.query(creat_sum_query, {'section': section, 'gid': gid, 'chunk_id': chunk_id})
    return s
    

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

def add_meta_sum(n4j, title, abstract, keyword, gid):
    """
    add_meta_sum(self.n4j, title, abstract, keyword, gid)
    为文档创建摘要节点并建立关系
    Args:
        n4j: Neo4j数据库连接对象
        title: 文档标题
        abstract: 文档摘要
        keyword: 文档关键词
        gid: 组ID
    Returns:
        object: 创建的摘要节点
    """
    creat_sum_query = """
        CREATE (s:Meta {title: $title, abstract: $abstract, keyword: $keyword, gid: $gid})
        RETURN s
        """
    s = n4j.query(creat_sum_query, {'title': title, 'abstract': abstract, 'keyword': keyword, 'gid': gid})

    link_sum_query = """
        MATCH (s:Meta {gid: $gid}), (n)
        WHERE n.gid = s.gid AND n:Summary
        CREATE (s)-[:basic_information]->(n)
        RETURN s, n
        """
    n4j.query(link_sum_query, {'gid': gid})

    # 所有chunk节点与Meta节点建立original_text relationship
    original_text_query = """
    MATCH (s:chunk {gid: $gid}), (n)
    WHERE n.gid = s.gid AND n:Meta
    CREATE (s)-[:original_text]->(n)
    RETURN s, n
    """
    n4j.query(original_text_query, {'gid': gid})   
    
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
    Returns:
        str: 生成的回答
    """

    selfcont = ret_context(n4j, gid)
    # selfsum = selfsum_context(n4j, gid)
    linkcont = link_context(n4j, gid)
    # user_one = "the question is: " + query + "the provided information is:" +  "".join(selfcont) + "the medical indexs are: " + selfsum
    user_one = "the question is: " + query + "the provided information is:" +  "".join(selfcont)
    res = call_llm(sys_prompt_one, user_one)
    user_two = "the question is: " + query + "the last response of it is:" + res + "the references are: " + "".join(linkcont)
    res = call_llm(sys_prompt_two, user_two)
    return res

def get_top_k_response(n4j, gids, query, query_embeddings, k):
    """
    根据查询生成综合回答
    Args:
        n4j: Neo4j数据库连接对象
        gids: 一组ID（每个gid代表一片文献）
        query: 查询问题
    Returns:
        str: 生成的回答
    """

    cont_doc = []
    reference_triples = []
    meta_docs = []
    for query_embedding in query_embeddings:
        
        doc, meta_doc, refs = get_top_k_triples(n4j, gids, query_embedding, k)
        meta_docs += meta_doc
        cont_doc.extend(doc)
        reference_triples.extend(refs)
    # cont_doc to str
    cont_doc_str = "".join(cont_doc)
    # reference_triples to str
    reference_triples_str = "".join(reference_triples)
    user_one = "the question is: " + query + "the references are: " + cont_doc_str
    res = call_llm(quick_sys_prompt_one, user_one)
    user_two = "the question is: " + query + "the last response of it is:" + res + "the document references the content of the medical dictionary are: " + reference_triples_str
    res = call_llm(quick_sys_prompt_two, user_two)

    return res,meta_docs,reference_triples

"""
    根据科室限定子图
"""
def get_department_top_k_response(n4j, department, query, query_embeddings, k):
    """
    根据查询生成综合回答
    Args:
        n4j: Neo4j数据库连接对象
        gids: 一组ID（每个gid代表一片文献）
        query: 查询问题
    Returns:
        str: 生成的回答
    """

    cont_doc = []
    reference_triples = []
    meta_docs = []

    """
      Todo:修改为多线程并行
    """
    for query_embedding in query_embeddings:
        
        doc, meta_doc, refs = get_department_top_k_triples(n4j, department, query_embedding, k)
        meta_docs += meta_doc
        cont_doc.extend(doc)
        reference_triples.extend(refs)
    # cont_doc to str
    cont_doc_str = "".join(cont_doc)
    # reference_triples to str
    reference_triples_str = "".join(reference_triples)
    user_one = "the question is: " + query + "the references are: " + cont_doc_str
    res = call_llm(quick_sys_prompt_one, user_one)
    user_two = "the question is: " + query + "the last response of it is:" + res + "the document references the content of the medical dictionary are: " + reference_triples_str
    res = call_llm(quick_sys_prompt_two, user_two)

    return res,meta_docs,reference_triples

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
        node1 = r.get('NodeId1') or ""
        rel = r.get('relType') or ""
        node2 = r.get('NodeId2') or ""
        cont.append(node1 + rel + node2)
    return cont

# 假设已经有Neo4j连接对象n4j
def get_top_k_triples(n4j, gids, query_embedding, top_k=10):
    """
    获取与问题三元组最相关的文档三元组，并返回相关节点和引用关系
    
    Args:
        n4j: Neo4j数据库连接对象
        gids: 文档ID列表
        query_embedding: 问题三元组的embedding向量
        top_k: 返回结果数量
        
    Returns:
        dict: 包含三元组、节点和引用关系的字典
    """
    # 1. 获取相关三元组
    cont_doc = []
    meta_doc = []
    query = """
    // 匹配满足条件的边和相关节点
    MATCH (s)-[r]->(o)
    WHERE r.gid IN $gids AND r.embedding IS NOT NULL

    // 计算余弦相似度并排序
    WITH s, r, o, 
         gds.similarity.cosine(r.embedding, $query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT $top_k

    // 查询Meta节点的标题
    MATCH (meta:Meta)
    WHERE meta.gid = r.gid

    MATCH (c:chunk)
    WHERE c.chunk_id = s.chunk_id AND c.gid = r.gid

    // 返回结果
    RETURN s.name AS subject, 
           s.chunk_id AS subject_chunk_id,
           r.name AS relation, 
           o.name AS object, 
           r.gid AS document_id,
           c.section AS section,
           c.content AS chunk,
           meta.title AS document_title,
           similarity AS score
    """
    
    params = {
        "gids": gids,
        "query_embedding": query_embedding,
        "top_k": top_k
    }
    triples_result = n4j.query(query, params)
    # 处理查询结果，将三元组信息添加到cont_doc列表中
    ind = 0
    for triple in triples_result:
        ind += 1
        cont_doc.append(f"The Triple reference:{str(ind)}: (Document: {triple['document_title']}, {triple['subject']} {triple['relation']} {triple['object']}, Score: {triple['score']})")
        """
        结构化存储每个信息
        """
        
        meta_doc.append({
            "document_title": triple['document_title'],
            "chunk": triple['chunk'],
            "section": triple['section'],
            "triple": triple['subject'] + triple['relation'] + triple['object'],
            "score": triple['score']
        })

    import json
    print("meta_doc:\n" + json.dumps(meta_doc, ensure_ascii=False, indent=2))
    print("\n")


    # 2. 收集所有相关的gid和节点ID
    related_gids = set()
    related_node_ids = set()
    for triple in triples_result:
        related_gids.add(triple['document_id'])
        related_node_ids.add(triple['subject'])
        related_node_ids.add(triple['object'])

    # 3. 获取所有相关节点的引用关系三元组
    reference_triples = []
    for node_id in related_node_ids:
        reference_query = """
        // 查找与指定节点有REFERENCE关系的所有节点
        MATCH (n)-[r:REFERENCE]-(m)
        WHERE n.name = $node_id
        RETURN n.name AS subject, 
               m.name AS object,
               type(r) AS relation
        """
        ref_results = n4j.query(reference_query, {'node_id': node_id})
        ind = 0
        for r in ref_results:
            ind += 1
            reference_triples.append("Reference " + str(ind) + ": " + r["subject"] + "has the reference that" + r['object'] + r['relation'])
    # 将return语句移到循环外部
    print("reference_triples:"+"\n".join(reference_triples))

    return cont_doc, meta_doc, reference_triples

def get_department_top_k_triples(n4j, department, query_embedding, top_k=10):
    """
    获取与问题三元组最相关的文档三元组，并返回相关节点和引用关系
    
    Args:
        n4j: Neo4j数据库连接对象
        gids: 文档ID列表
        query_embedding: 问题三元组的embedding向量
        top_k: 返回结果数量
        
    Returns:
        dict: 包含三元组、节点和引用关系的字典
    """
    # 1. 获取相关三元组
    cont_doc = []
    meta_doc = []
    query = """
    // 匹配满足条件的边和相关节点
    MATCH (s)-[r]->(o)
    WHERE r.department = $department AND r.embedding IS NOT NULL

    // 计算余弦相似度并排序
    WITH s, r, o, 
         gds.similarity.cosine(r.embedding, $query_embedding) AS similarity
    ORDER BY similarity DESC
    LIMIT $top_k

    // 查询Meta节点的标题
    MATCH (meta:Meta)
    WHERE meta.gid = r.gid

    MATCH (c:chunk)
    WHERE c.chunk_id = s.chunk_id AND c.gid = r.gid

    // 返回结果
    RETURN s.name AS subject, 
           s.chunk_id AS subject_chunk_id,
           r.name AS relation, 
           o.name AS object, 
           r.gid AS document_id,
           c.section AS section,
           c.content AS chunk,
           meta.title AS document_title,
           similarity AS score
    """
    
    params = {
        "department": department,
        "query_embedding": query_embedding,
        "top_k": top_k
    }
    triples_result = n4j.query(query, params)
    # 处理查询结果，将三元组信息添加到cont_doc列表中
    ind = 0
    for triple in triples_result:
        ind += 1
        cont_doc.append(f"The Triple reference:{str(ind)}: (Document: {triple['document_title']}, {triple['subject']} {triple['relation']} {triple['object']}, Score: {triple['score']})")
        """
        结构化存储每个信息
        """
        
        meta_doc.append({
            "document_title": triple['document_title'],
            "chunk": triple['chunk'],
            "section": triple['section'],
            "triple": triple['subject'] + triple['relation'] + triple['object'],
            "score": triple['score']
        })

    import json
    print("meta_doc:\n" + json.dumps(meta_doc, ensure_ascii=False, indent=2))
    print("\n")


    # 2. 收集所有相关的gid和节点ID
    related_gids = set()
    related_node_ids = set()
    for triple in triples_result:
        related_gids.add(triple['document_id'])
        related_node_ids.add(triple['subject'])
        related_node_ids.add(triple['object'])

    # 3. 获取所有相关节点的引用关系三元组
    reference_triples = []
    for node_id in related_node_ids:
        reference_query = """
        // 查找与指定节点有REFERENCE关系的所有节点
        MATCH (n)-[r:REFERENCE]-(m)
        WHERE n.name = $node_id
        RETURN n.name AS subject, 
               m.name AS object,
               type(r) AS relation
        """
        ref_results = n4j.query(reference_query, {'node_id': node_id})
        ind = 0
        for r in ref_results:
            ind += 1
            reference_triples.append("Reference " + str(ind) + ": " + r["subject"] + "has the reference that" + r['object'] + r['relation'])
    # 将return语句移到循环外部
    print("reference_triples:"+"\n".join(reference_triples))

    return cont_doc, meta_doc, reference_triples

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
            WHERE NOT n:Summary AND NOT m:Summary AND NOT n:chunk AND NOT m:chunk AND n.gid = m.gid AND n.gid = $gid AND n<>m AND apoc.coll.sort(labels(n)) = apoc.coll.sort(labels(m))
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

def edges_embedding(n4j, gid):
    """
    为gid下的所有边构建的三元组生成embedding，并存储在边的embedding属性中
    Args:
        n4j: Neo4j数据库连接对象
        gid: 组ID
    Returns:
        object: 构建embedding的结果
    """
    # 1. 查询指定gid下的所有边及其相关节点
    query = """
    MATCH (s)-[r]->(o)
    WHERE r.gid = $gid
    RETURN id(r) AS rel_id, s.name AS subject, r.name AS relation, o.name AS object
    """
    
    edges = n4j.query(query, {'gid': gid})
    
    # 2. 为每条边构建三元组文本并生成embedding
    update_count = 0
    for edge in edges:
        # 构建三元组文本
        triple_text = f"{edge['subject']} {edge['relation']} {edge['object']}"
        
        # 使用get_embedding函数生成embedding
        embedding = get_embedding(triple_text)
        
        # 3. 将embedding存储到边的属性中
        update_query = """
        MATCH ()-[r]->() 
        WHERE id(r) = $rel_id
        SET r.embedding = $embedding
        """
        
        n4j.query(update_query, {'rel_id': edge['rel_id'], 'embedding': embedding})
        update_count += 1
    
    return {
        'status': 'success',
        'message': f'成功为 {update_count} 条边生成并存储embedding',
        'gid': gid
    }

def sums_embedding(n4j, gid):
    """
    为gid下的Summary节点生成embedding，并存储在节点的embedding属性中
    Args:
        n4j: Neo4j数据库连接对象
        gid: 组ID
    Returns:
        object: 构建embedding的结果
    """
    # 1. 查询指定gid下的所有Summary节点及其相关节点
    query = """
    MATCH (s:Summary)   
    WHERE s.gid = $gid
    RETURN s.id AS summary_id, s.content AS summary_content
    """

    summaries = n4j.query(query, {'gid': gid})
    
    sum_embedding = get_embedding(summaries[0]['summary_content'])

    query = """
    MATCH (s:Summary)
    WHERE s.gid = $gid
    SET s.embedding = $embedding
    """
    n4j.query(query, {'gid': gid, 'embedding': sum_embedding})
    return {
        'status':'success',
    }

    # 1. 查询指定gid下的所有节点

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
    print("建立两个图之间的引用关系(余弦相似度)")
    trinity_query =  """
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
        WITH n, m, 0.95 AS threshold

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
    print(f"建立了 {len(result)} 个引用关系")
    return result

def str_uuid():
    """
    生成UUID字符串
    Returns:
        str: 生成的UUID字符串
    """
    generated_uuid = uuid.uuid4()
    return str(generated_uuid)

def check_node_exists(n4j, node_id):
    """
    检查节点是否已存在于数据库中
    Args:
        n4j: Neo4j数据库连接对象
        node_id: 节点ID
    Returns:
        bool: 节点是否存在
    """
    query = "MATCH (n) WHERE n.id = $node_id RETURN count(n) as count"
    result = n4j.query(query, params={"node_id": node_id})
    return result[0]["count"] > 0

def get_existing_nodes_by_gid(n4j, gid):
    """
    获取指定gid的所有节点ID
    Args:
        n4j: Neo4j数据库连接对象
        gid: 组ID
    Returns:
        list: 节点ID列表
    """
    query = "MATCH (n) WHERE n.gid = $gid RETURN n.id as id"
    result = n4j.query(query, params={"gid": gid})
    return [record["id"] for record in result if record["id"]]


