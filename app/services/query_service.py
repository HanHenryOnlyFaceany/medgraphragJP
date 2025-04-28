from tokenize import Triple
from nltk.parse.featurechart import sent
from core.config import settings
from src.dataloader import load_high
from src.summerize import process_chunks
from src.retrieve import *
from src.utils.util import *
from camel.storages import Neo4jGraph
from services.graph_service import *
from src.utils.util import *

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
    
    """
    三元组的抽取+embedding 6-7s
    summary的抽取 4s
    embedding+summary的检索 ms级别
    三元组的检索 2s以内
    大模型调用（两次迭代调用，每次token大概200-300） 
    """

    def quick_query(self, query_text, query_embeddings, k=5):
        """执行知识图谱查询
        
        Args:
            query_text: 查询文本
            gid: 图元素组ID
            
        Returns:
            dict: 包含查询结果的字典
        """
        # question = load_high(query_text)
        # 通过LLM生成结构性summary并通过embedding获取gid
        sum = process_chunks(query_text)
        gids, top_scores = quick_ret_top_k(self.n4j, sum, k)
        # 使用现有的get_response函数获取回答
        answer,meta_doc,reference_triples = get_top_k_response(self.n4j, gids, query_text, query_embeddings, k)
        
        # 获取引用的参考资料
        # references = self._get_references(gid)
        
        return answer, meta_doc, reference_triples

    def quick_query_by_department(self, query_text, query_embeddings, k=5):
        """执行知识图谱查询
        
        Args:
            query_text: 查询文本
            gid: 图元素组ID
            
        Returns:
            dict: 包含查询结果的字典
        """
        # question = load_high(query_text)
        # 通过LLM生成结构性summary并通过embedding获取gid
        # sum = process_chunks(query_text)
        # gids, top_scores = quick_ret_top_k(self.n4j, sum, k)
        """
        通过MCP方法根据query_txt指定department(疾病所属的类别)
        get_department为一个MCP方法，需要根据query_txt指定department
        """
        departments = [
            "内科-内分泌科", "内科-消化内科", "内科-呼吸内科", "内科-神经内科", "内科-普通内科","内科-心血管内科", "内科-老年病科", "内科-风湿免疫科", "内科-血液科", "内科-肾病内科","内科-感染内科", "中医科-中医男科", "中医科-中医内科", "中医科-中医儿科", "中医科-中医皮肤科",
            "中医科-中医骨科", "中医科-中医呼吸科", "中医科-中医妇产科", "中医科-中医消化科", "中医科-中医外科","中医科-中医眼科", "中医科-中医肛肠科", "中医科-中医心身医学科", "中医科-中医神经内科", "中医科-中医肝胆科","中医科-中医肿瘤科", "中医科-中医心内科", "中医科-针灸推拿科", "中医科-中医内分泌科", "中医科-中医乳腺科","中医科-中医治未病科", "中医科-中医耳鼻喉科", "中医科-中医康复科", "中医科-中医肾病科", "中医科-中医周围血管病科","中医科-中医泌尿科", "中医科-中医风湿免疫科", "中医科-中医血液科", "中医科-中医老年病科", "皮肤性病科",
            "皮肤性病科-皮肤科", "皮肤性病科-性病科", "皮肤性病科-小儿皮肤科", 
            "儿科-小儿内科", "儿科-小儿呼吸科","儿科-小儿消化科", "儿科-新生儿科", "儿科-小儿神经内科", "儿科-小儿血液科", "儿科-小儿心脏科","儿科-小儿内分泌科", "儿科-小儿免疫科", "妇产科-妇科", "妇产科-产科", "妇产科-生殖健康与不孕症科",
            "妇产科-计划生育科", 
            "外科-泌尿外科", "外科-普通外科", "外科-肛肠科", "外科-乳腺外科","外科-神经外科", "外科-胸外科", "外科-肝胆外科", "外科-心血管外科", "外科-整形外科","外科-烧伤科", "外科-甲状腺外科", "外科-胃肠外科", "外科-血管外科", 
            "男科",
            "骨科-足踝外科", "骨科-创伤科", "骨科-关节科", "骨科-脊柱科", "骨科-手外科","骨科-骨科", 
            "精神心理科-临床心理科", "精神心理科-精神科", "精神心理科-青少年儿童心理科", "精神心理科-成瘾医学科",
            "营养科", 
            "耳鼻咽喉头颈外科", "耳鼻咽喉头颈外科-耳鼻咽喉科", "耳鼻咽喉头颈外科-头颈外科", "耳鼻咽喉头颈外科-小儿耳鼻咽喉科",
            "眼科-眼科", "眼科-眼底科", "眼科-视光中心", "眼科-角膜科", "眼科-青光眼科","眼科-白内障科", "眼科-小儿眼科", "眼科-眼整形科", "眼科-眼外伤科", "口腔科-口腔综合科",
            "口腔科-牙周科", "口腔科-颌面外科", "口腔科-牙体牙髓科", "口腔科-口腔修复科", "口腔科-儿童口腔科","口腔科-种植科", "口腔科-正畸科", "口腔科-口腔黏膜科", "小儿外科-小儿外科", "小儿外科-小儿骨科",
            "小儿外科-小儿泌尿科", "小儿外科-小儿心胸外科", "小儿外科-小儿神经外科", 
            "儿童保健科-小儿保健科", "儿童保健科-小儿康复科",
            "肿瘤科-肿瘤内科", "肿瘤科-肿瘤外科", "肿瘤科-肿瘤妇科", "肿瘤科-骨肿瘤科", "肿瘤科-介入与放疗中心",
            "中西医结合科", 
            "预防保健科", 
            "医学影像科", "医学影像科-放射科", "医学影像科-超声科","医学影像科-核医学科", 
            "医疗美容科", 
            "康复医学科", 
            "药学门诊", 
            "疼痛科",
            "传染科", "传染科-肝病科", "病理科", "急诊科", "麻醉科","运动医学科"
        ]
        """Todo"""
        department = get_department(query_text)
        # 使用现有的get_response函数获取回答
        answer,meta_doc,reference_triples = get_department_top_k_response(self.n4j, department, query_text, query_embeddings, k)
        
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

    """
    sentences: 句子列表
    多线程执行get_extraction_result方法，结果存储在query_embeddings
    """
    def get_querys_embedding(self, sentences):
        # 导入必要的模块
        import concurrent.futures
        import logging
        import os
        
        # 创建logger（如果项目中已有logger配置，可以删除这部分）
        logger = logging.getLogger(__name__)
        
        query_embeddings = []
        max_workers = min(32, (os.cpu_count() or 4) * 2)

        # 健壮性检查：确保sentences不为空且是列表类型
        if not sentences or not isinstance(sentences, list):
            logger.warning("输入的sentences为空或不是列表类型")
            return []

        try:
            # 使用线程池并行处理所有句子
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务到线程池
                future_to_sentence = {
                    executor.submit(self.graph_service.get_extraction_result, sentence): sentence 
                    for sentence in sentences
                }
                
                # 收集所有三元组结果
                all_triples = []
                for future in concurrent.futures.as_completed(future_to_sentence):
                    sentence = future_to_sentence[future]
                    try:
                        result = future.result()
                        if result and isinstance(result, dict) and 'triple_list' in result:
                            all_triples.append(result)
                    except Exception as e:
                        logger.error(f"处理句子 '{sentence}' 时出错: {e}")
                        # 对于失败的句子，使用原始句子生成embedding
                        embedding = get_embedding(sentence)
                        query_embeddings.append(embedding)
                
                # 如果没有成功提取出三元组，则对每个原始句子生成embedding
                if not all_triples:
                    for sentence in sentences:
                        embedding = get_embedding(sentence)
                        query_embeddings.append(embedding)
                else:
                    # 处理所有成功提取的三元组
                    for triple_result in all_triples:
                        for edge in triple_result.get('triple_list', []):
                            # 构建三元组文本
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
                            
                            # 生成embedding
                            embedding = get_embedding(triple_text)
                            query_embeddings.append(embedding)
        except Exception as e:
            logger.error(f"在get_querys_embedding中执行线程池时出错: {e}")
            # 出错时的备选方案：对每个原始句子生成embedding
            for sentence in sentences:
                try:
                    embedding = get_embedding(sentence)
                    query_embeddings.append(embedding)
                except Exception as inner_e:
                    logger.error(f"为句子 '{sentence}' 生成embedding时出错: {inner_e}")

        return query_embeddings


    def _get_references(self, gid):
        """获取指定gid的引用资料"""
        # query = """
        # MATCH (n {gid: $gid})-[:REFERENCE]->(m)
        # RETURN DISTINCT m.id as reference
        # """
        # result = self.n4j.query(query, {"gid": gid})
        # return [record["reference"] for record in result if record["reference"]]