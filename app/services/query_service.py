from core.config import settings
from src.dataloader import load_high
from src.summerize import process_chunks
from src.retrieve import *
from src.utils.util import *
from camel.storages import Neo4jGraph
from services.graph_service import *

class QueryService:
    def __init__(self, n4j):
        self.n4j = Neo4jGraph(
            url=settings.NEO4J_URL,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )
        self.graph_service = GraphService()


    def precise_query(self, query_text):
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
        
        # 获取引用的参考资料
        references = self._get_references(gid)
        
        return {
            "answer": answer,
            "references": references
        }

    def quick_query(self, query_text, query_embeddings, k=5):
        """执行知识图谱查询
        
        Args:
            query_text: 查询文本
            gid: 图元素组ID
            
        Returns:
            dict: 包含查询结果的字典
        """
        # question = load_high(query_text)
        sum = process_chunks(query_text)
        gids, top_scores = quick_ret_top_k(self.n4j, sum, k)
        # 使用现有的get_response函数获取回答
        answer,meta_doc,reference_triples = get_top_k_response(self.n4j, gids, query_text, query_embeddings, k)
        
        # 获取引用的参考资料
        # references = self._get_references(gid)
        
        return answer, meta_doc, reference_triples

    def get_query_embedding(self, query_text):
        # 调用 GraphService 的方法
        # query_triples = {}
        query_triples = self.graph_service.get_extraction_result(query_text)
        query_embeddings = []

        # 判断query_triples是否为空
        if not query_triples:
            embedding = get_embedding(query_text)
            # 获取query的嵌入向量
            query_embeddings.append(embedding)
        else:
            for edge in query_triples.get('triple_list',[]):
                # 构建三元组文本，判断是否存在
                triple_text = ""
                if edge.get('head'):
                    triple_text = f"{edge['head']} "
                if edge.get('relation'):
                    triple_text += f"{edge['relation']} "
                if edge.get('tail'):
                    triple_text += f"{edge['tail']}"
                
                # 确保三元组文本不为空
                if not triple_text.strip():
                    continue
                
                # 使用get_embedding函数生成embedding
                embedding = get_embedding(triple_text)
                # 获取query的嵌入向量
                query_embeddings.append(embedding)

        return query_embeddings


    def _get_references(self, gid):
        """获取指定gid的引用资料"""
        # query = """
        # MATCH (n {gid: $gid})-[:REFERENCE]->(m)
        # RETURN DISTINCT m.id as reference
        # """
        # result = self.n4j.query(query, {"gid": gid})
        # return [record["reference"] for record in result if record["reference"]]