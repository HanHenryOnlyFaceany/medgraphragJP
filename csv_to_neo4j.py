import pandas as pd
import uuid
from neo4j import GraphDatabase
from utils import get_embedding, str_uuid
import os
import time


def load_csv_to_neo4j(csv_path, n4j, batch_size=1000):
    """
    高效地将CSV文件导入到Neo4j数据库中
    
    Args:
        csv_path: CSV文件路径
        n4j: Neo4j数据库连接对象
        batch_size: 批处理大小，默认1000条记录一批
    
    Returns:
        str: 导入的图的组ID (gid)
    """
    # 读取CSV文件
    print(f"开始读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)


    # 验证CSV格式
    required_columns = ['head_entity', 'head_type', 'relation', 'tail_entity', 'tail_type']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必要的列: {col}")
    
    # 为整个图分配一个唯一的组ID
    gid = str_uuid()
    print(f"为此图分配的组ID: {gid}")
    

    # 清理 NaN 值
    df = df.dropna()
    print(f"清理 NaN 值后剩余 {len(df)} 条记录")


    # 获取所有唯一实体
    head_entities = df[['head_entity', 'head_type']].drop_duplicates()
    tail_entities = df[['tail_entity', 'tail_type']].drop_duplicates()
    
    # 合并所有唯一实体
    entities = pd.concat([
        head_entities.rename(columns={'head_entity': 'entity', 'head_type': 'type'}),
        tail_entities.rename(columns={'tail_entity': 'entity', 'tail_type': 'type'})
    ]).drop_duplicates()
    
    # 创建实体节点
    total_entities = len(entities)
    print(f"开始创建 {total_entities} 个实体节点...")
    
    # 再次确保没有 NaN 值
    entities = entities.dropna()


    # 批量创建节点
    start_time = time.time()
    for i in range(0, total_entities, batch_size):
        batch = entities.iloc[i:i+batch_size]
        
        # 构建批量创建节点的Cypher查询
        create_nodes_query = """
        UNWIND $nodes AS node
        MERGE (n:`$label` {id: node.entity, gid: $gid})
        SET n.type = node.type
        RETURN count(n) as created_count
        """
        
        # 按类型分组批量创建节点
        for entity_type, group in batch.groupby('type'):
            nodes_data = [{'entity': row['entity']} for _, row in group.iterrows()]
            if nodes_data:
                result = n4j.query(
                    create_nodes_query.replace('$label', entity_type),
                    {'nodes': nodes_data, 'gid': gid}
                )
                
        # 显示进度
        progress = min(i + batch_size, total_entities) / total_entities * 100
        print(f"节点创建进度: {progress:.2f}% ({min(i + batch_size, total_entities)}/{total_entities})")
    
    node_creation_time = time.time() - start_time
    print(f"节点创建完成，耗时: {node_creation_time:.2f}秒")
    
    # 创建关系
    total_relations = len(df)
    print(f"开始创建 {total_relations} 个关系...")
    
    # 批量创建关系
    start_time = time.time()
    for i in range(0, total_relations, batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # 构建批量创建关系的Cypher查询
        create_relations_query = """
        UNWIND $relations AS rel
        MATCH (a {id: rel.head_entity, gid: $gid})
        MATCH (b {id: rel.tail_entity, gid: $gid})
        MERGE (a)-[r:`$relation_type` {gid: $gid}]->(b)
        RETURN count(r) as created_count
        """
        
        # 按关系类型分组批量创建关系
        for relation_type, group in batch.groupby('relation'):
            relations_data = [
                {
                    'head_entity': row['head_entity'],
                    'tail_entity': row['tail_entity']
                } for _, row in group.iterrows()
            ]
            if relations_data:
                result = n4j.query(
                    create_relations_query.replace('$relation_type', relation_type),
                    {'relations': relations_data, 'gid': gid}
                )
        
        # 显示进度
        progress = min(i + batch_size, total_relations) / total_relations * 100
        print(f"关系创建进度: {progress:.2f}% ({min(i + batch_size, total_relations)}/{total_relations})")
    
    relation_creation_time = time.time() - start_time
    print(f"关系创建完成，耗时: {relation_creation_time:.2f}秒")
    
    return gid


def add_embeddings_batch(n4j, gid, batch_size=100):
    """
    批量为指定gid的所有节点添加嵌入向量
    
    Args:
        n4j: Neo4j数据库连接对象
        gid: 组ID
        batch_size: 批处理大小
    """
    # 获取所有需要添加embedding的节点
    query = """
    MATCH (n)
    WHERE n.gid = $gid AND NOT EXISTS(n.embedding)
    RETURN n.id AS id
    LIMIT $batch_size
    """
    
    print(f"开始为组ID {gid} 的节点添加嵌入向量...")
    start_time = time.time()
    
    # 循环处理直到没有更多节点需要处理
    total_processed = 0
    while True:
        nodes = n4j.query(query, {'gid': gid, 'batch_size': batch_size})
        if not nodes:
            break
            
        for node in nodes:
            if node['id']:
                # 这里调用utils.py中的get_embedding函数获取嵌入向量
                # 注意：根据要求，这里不需要实现获取embedding的逻辑
                embedding = get_embedding(node['id'])
                
                # 更新节点的embedding属性
                update_query = "MATCH (n) WHERE n.id = $node_id AND n.gid = $gid SET n.embedding = $embedding"
                n4j.query(update_query, {'node_id': node['id'], 'gid': gid, 'embedding': embedding})
                
        total_processed += len(nodes)
        print(f"已处理 {total_processed} 个节点的嵌入向量")
    
    total_time = time.time() - start_time
    print(f"嵌入向量添加完成，总共处理了 {total_processed} 个节点，耗时: {total_time:.2f}秒")


def main(csv_path, neo4j_url, neo4j_user, neo4j_password, add_embeddings=True):
    """
    主函数，执行CSV导入Neo4j的完整流程
    
    Args:
        csv_path: CSV文件路径
        neo4j_uri: Neo4j数据库URI
        neo4j_user: Neo4j用户名
        neo4j_password: Neo4j密码
        add_embeddings: 是否添加嵌入向量，默认为True
    """
    # 连接Neo4j数据库
    from camel.storages import Neo4jGraph
    n4j = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_password)
    
    try:
        # 导入CSV数据到Neo4j
        start_time = time.time()
        gid = load_csv_to_neo4j(csv_path, n4j)
        import_time = time.time() - start_time
        print(f"CSV导入完成，总耗时: {import_time:.2f}秒")
        
        # 如果需要，添加嵌入向量
        if add_embeddings:
            add_embeddings_batch(n4j, gid)
        
        print(f"导入完成！图的组ID为: {gid}")
        return gid
    except Exception as e:
        print(f"导入过程中发生错误: {str(e)}")
        raise
    finally:
        # 关闭Neo4j连接
        if hasattr(n4j, 'close'):
            n4j.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将CSV文件导入Neo4j数据库')
    parser.add_argument('--csv', required=True, help='CSV文件路径', default='./dataset_cn/HITCPubMedKg_2/HITCPubMed-KGv2_0_mini.csv')
    parser.add_argument('--url', default='bolt://localhost:7687', help='Neo4j数据库URI')
    parser.add_argument('--user', default='neo4j', help='Neo4j用户名')
    parser.add_argument('--password', default='zcx1264521752', required=True, help='Neo4j密码')
    
    args = parser.parse_args()
    
    main(
        csv_path=args.csv,
        neo4j_url=args.url,
        neo4j_user=args.user,
        neo4j_password=args.password,
    )