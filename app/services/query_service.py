from core.config import settings
from src.dataloader import load_high
from src.summerize import process_chunks
from src.retrieve import seq_ret
from src.utils import *
from camel.storages import Neo4jGraph

class QueryService:
    def __init__(self, n4j):
        self.n4j = Neo4jGraph(
            url=settings.NEO4J_URL,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )
    
    def query(self, query_text):
        """执行知识图谱查询
        
        Args:
            query_text: 查询文本
            gid: 图元素组ID
            
        Returns:
            dict: 包含查询结果的字典
        """
        # question = load_high(query_text)
        sum = process_chunks(query_text)
        gid = seq_ret(self.n4j, sum)
        # 使用现有的get_response函数获取回答
        answer = get_response(self.n4j, gid, query_text)
        
        # # 获取引用的参考资料
        # references = self._get_references(gid)
        
        # return {
        #     "answer": answer,
        #     # "references": references
        # }

        return answer
    
    def _get_references(self, gid):
        """获取指定gid的引用资料"""
        query = """
        MATCH (n {gid: $gid})-[:REFERENCE]->(m)
        RETURN DISTINCT m.id as reference
        """
        result = self.n4j.query(query, {"gid": gid})
        return [record["reference"] for record in result if record["reference"]]