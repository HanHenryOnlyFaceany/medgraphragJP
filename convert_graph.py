import uuid

def parse_line(line):
    # 解析一行，返回 (head, head_type, relation, tail, tail_type)
    parts = line.strip().split('\t')
    if len(parts) != 3:
        return None
    head, relation, tail = parts
    head_entity, head_type = head.split('@@')
    tail_entity, tail_type = tail.split('@@')
    return head_entity, head_type, relation, tail_entity, tail_type

def add_triple_to_neo4jgraph(n4j: 'Neo4jGraph', head, head_type, tail, tail_type, relation, gid):
    # 使用Neo4jGraph的query方法插入三元组
    cypher = f"""
        MERGE (h:{head_type} {{name: $head}})
        SET h.gid = $gid
        MERGE (t:{tail_type} {{name: $tail}})
        SET t.gid = $gid
        MERGE (h)-[r:`{relation}`]->(t)
        SET r.gid = $gid
    """
    n4j.query(cypher, {"head": head, "tail": tail, "gid": gid})

def process_txt_to_neo4jgraph(txt_path, n4j: 'Neo4jGraph'):
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '@@' not in line or '\t' not in line:
                continue
            parsed = parse_line(line)
            if not parsed:
                continue
            head, head_type, relation, tail, tail_type = parsed
            gid = str(uuid.uuid4())
            add_triple_to_neo4jgraph(n4j, head, head_type, tail, tail_type, relation, gid)

# 用法示例
if __name__ == "__main__":
    from camel.storages import Neo4jGraph
    n4j = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="your_password"
    )
    txt_path = "your_triple.txt"
    process_txt_to_neo4jgraph(txt_path, n4j)