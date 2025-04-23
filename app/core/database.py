from camel.storages import Neo4jGraph
from core.config import settings

def get_neo4j_connection():
    """获取Neo4j数据库连接"""
    return Neo4jGraph(
        url=settings.NEO4J_URL,
        username=settings.NEO4J_USERNAME,
        password=settings.NEO4J_PASSWORD
    )