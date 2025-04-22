import os
from camel.storages import Neo4jGraph
from utils import ref_link

def main():
    """
    单独执行ref_link方法，建立两个图之间的引用关系
    """
    # 连接Neo4j数据库
    url = os.getenv("NEO4J_URL")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    n4j = Neo4jGraph(
        url=url,
        username=username,
        password=password
    )
    
    # 定义要连接的两个图的GID
    gid = "f83b4c4c-cb83-48c3-9fad-3fa32df65041"  # 源图GID
    trinity_gid1 = "e0b1ebf5-984b-4310-bece-cbc7f6c9eb42"  # 目标图GID
    
    print(f"正在建立图 {gid} 和图 {trinity_gid1} 之间的引用关系...")
    
    # 执行ref_link方法
    result = ref_link(n4j, gid, trinity_gid1)
    
    # 输出结果
    print(f"引用关系建立完成，共建立 {len(result)} 个关系")
    for i, r in enumerate(result):
        print(f"关系 {i+1}: {r}")

if __name__ == "__main__":
    main()