import os
import uuid
from camel.storages import Neo4jGraph
# from camel.agents import KnowledgeGraphAgent
# from camel.loaders import UnstructuredIO
# from camel.models import ModelFactory
# from camel.types import ModelPlatformType
# from camel.configs import ChatGPTConfig
from unstructured.documents.elements import Title
from core.config import settings
from src.utils.util import *
from src.utils import *
from src.data_chunk import *
from src.models import *
from src.pipeline import *
import argparse
import os
from src.modules import *
import nltk
nltk.download('punkt')

# model configuration



class GraphService:
    def __init__(self):
        self.n4j = Neo4jGraph(
            url=settings.NEO4J_URL,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD
        )

        self.model = LocalServer(model_name_or_path="qwen2.5-32b-instruct")

        # Load configuration
        config = load_extraction_config("/Users/hanhenry99/jianpei/medgraphragJP/app/src/examples/config/Triple2KG.yaml")
        # Model config
        self.pipeline = Pipeline(self.model)
        self.extraction_config = config['extraction']
        self.construct_config = config['construct']
        self.target_gid = "92a31f57-2d5e-4cda-bcc4-ca5c483cde04"
    
    def create_graph(self, file_path, grained_chunk=True, ingraphmerge=True):
        """创建知识图谱
        
        Args:
            content: 文本内容
            grained_chunk: 是否使用细粒度分块
            ingraphmerge: 是否合并相似节点
            
        Returns:
            dict: 包含图信息的字典
        """
        file_path = os.path.join(args.data_path, file_name)
        content = load_high(file_path)
        # 生成唯一的组ID
        gid = str(uuid.uuid4())
        
        # 处理内容块
        if grained_chunk:
            from data_chunk import run_chunk
            content_chunks = run_chunk(content, gid)
        else:
            content_chunks = [content]
        


        for i, cont in enumerate(content_chunks, 1):
            try:
                # 创建元素
                element_example = self.uio.create_element_from_text(text=cont)
                
                # 生成知识图谱元素
                graph_elements = self.kg_agent.run(element_example, parse_graph_elements=True)
                
                # 添加嵌入向量
                graph_elements = add_ge_emb(graph_elements)
                
                # 添加组ID
                graph_elements = add_gid(graph_elements, gid)
                
                # 将图元素添加到数据库
                self.n4j.add_graph_elements(graph_elements=[graph_elements])
                processed_count += 1
                
            except Exception as e:
                print(f"处理块 {i} 时出错: {e}")
        
        # 合并相似节点
        if ingraphmerge:
            merge_similar_nodes(self.n4j, gid)
        
        # 生成内容摘要
        add_sum(self.n4j, content, gid)
        
        # 获取节点和关系数量
        node_count = self._get_node_count(gid)
        relationship_count = self._get_relationship_count(gid)
        
        return {
            "gid": gid,
            "node_count": node_count,
            "relationship_count": relationship_count,
            "message": f"成功处理 {processed_count}/{len(content_chunks)} 个内容块"
        }
    

    def build_docs_graph(self, file_path, grained_chunk=True, ingraphmerge=True):
        """创建知识图谱,针对于论文数据
        Todo：
            如果是报告，txt数据，根据提示词生成直接生成对应的Json数据
            如果没有title，abstract，keyword，直接生成
        
        Args:
            json_content: JSON格式的文本内容
            grained_chunk: 是否使用细粒度分块
            ingraphmerge: 是否合并相似节点
            
        Returns:
            dict: 包含图信息的字典
        """
        
        # 将file_path下的json文件进行解析
        data = load_json(file_path)

        title = data[0].get("title")
        abstract = [item["content"] for item in data if item.get("type") == "ABSTRACT"]
        keyword = [item["content"] for item in data if item.get("type") == "KEYWORDS"]

        paragraph_data = [item for item in data if item.get("type") in ["PARAGRAPH", "TABLE"]]


        # 生成唯一的组ID
        gid = str(uuid.uuid4())
        
        # 处理内容块
        if grained_chunk:
            content_chunks = run_docs_chunk(paragraph_data, gid)

        for key, chunk in content_chunks.items():
            try:
                para = " ".join([x for x in chunk['propositions']])

                # 调用新方法获取提取结果
                frontend_res = self.get_extraction_result(para)

                # 添加组ID
                # graph_elements = add_gid(graph_elements, gid)

                extraction_result = json.dumps(frontend_res, indent=2)

                chunk_id = chunk['chunk_id']
                
                # 构造知识图谱并添加gid和chunk_id
                construct_kg(self.construct_config, extraction_result, gid, chunk_id)

                # 添加嵌入向量
                add_all_embeddings(self.construct_config, gid)

                # 添加chunk节点
                add_chunk(self.n4j, gid, chunk['chunk_id'], chunk['content'])

                # 为chunk添加section属性
                add_section(self.n4j, gid, chunk['chunk_id'], chunk['section_title'])

                
                # 将图元素添加到数据库
                # self.n4j.add_graph_elements(graph_elements=[graph_elements])
                
            except Exception as e:
                print(f"处理块 {i} 时出错: {e}")
        
        # 合并相似节点
        if ingraphmerge:
            merge_similar_nodes(self.n4j, gid)

        # 添加三元组的embedding
        edges_embedding(self.n4j, gid)
        
        content = " ".join([x['content'] for x in paragraph_data])
        # 生成内容摘要
        add_sum(self.n4j, content, gid)

        sums_embedding(self.n4j, gid)

        # 添加文档元数据节点并与summary以及所有chunk节点建立关系
        add_meta_sum(self.n4j, title, abstract, keyword, gid)

        self.create_reference_links(gid, self.target_gid)
        
        return {
            "gid": gid,
            "message": f"成功处理"
        }
    
    def create_reference_links(self, source_gid, target_gid):
        """创建两个图之间的引用关系
        
        Args:
            source_gid: 源图GID
            target_gid: 目标图GID
            
        Returns:
            dict: 包含引用关系信息的字典
        """
        result = ref_link(self.n4j, source_gid, target_gid)
        
        return {
            "source_gid": source_gid,
            "target_gid": target_gid,
            "link_count": len(result),
            "links": result
        }
    
    
    def _get_node_count(self, gid):
        """获取指定gid的节点数量"""
        query = "MATCH (n) WHERE n.gid = $gid RETURN count(n) as count"
        result = self.n4j.query(query, {"gid": gid})
        return result[0]["count"] if result else 0
    
    def get_extraction_result(self, text):
        """
        获取文本的提取结果
        
        Args:
            text: 待提取的文本内容
            
        Returns:
            dict: 提取结果
        """
        return self.pipeline.get_extract_result(
            task=self.extraction_config['task'], 
            instruction=self.extraction_config['instruction'], 
            text=text, 
            output_schema=self.extraction_config['output_schema'], 
            constraint=self.extraction_config['constraint'], 
            truth=self.extraction_config['truth'], 
            mode=self.extraction_config['mode'], 
            update_case=self.extraction_config['update_case'], 
            show_trajectory=self.extraction_config['show_trajectory'],
            construct=self.construct_config, 
        )
    def delete_graph_by_gid(self, gid):
        """
        删除指定gid的所有节点和关系
        
        Args:
            gid: 要删除的图的组ID
            
        Returns:
            dict: 包含删除结果的字典
        """
        query = "MATCH (n) WHERE n.gid = $gid DETACH DELETE n"
        result = self.n4j.query(query, {"gid": gid})
        return {
            "message": f"成功删除GID为{gid}的图",
            "result": result
        }
    
    def clean_all_graphs(self, exclude_gid="92a31f57-2d5e-4cda-bcc4-ca5c483cde04"):
        """
        删除除了指定gid外的所有图
        
        Args:
            exclude_gid: 要排除的图的组ID，默认为医学词典图谱ID
            
        Returns:
            dict: 包含删除结果的字典
        """
        query = f"MATCH (n) WHERE n.gid <> $exclude_gid DETACH DELETE n"
        result = self.n4j.query(query, {"exclude_gid": exclude_gid})
        return {
            "message": f"成功清理所有图（保留GID为{exclude_gid}的图）",
            "result": result
        }
    
    def _get_relationship_count(self, gid):
        """获取指定gid的关系数量"""
        query = "MATCH ()-[r]-() WHERE r.gid = $gid RETURN count(r) as count"
        result = self.n4j.query(query, {"gid": gid})
        return result[0]["count"] if result else 0
    
    