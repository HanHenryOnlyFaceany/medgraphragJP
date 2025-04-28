from src.utils.util import *

sys_p = """
Assess the similarity of the two provided summaries and return a rating from these options: 'very similar', 'similar', 'general', 'not similar', 'totally not similar'. Provide only the rating.
"""

def seq_ret(n4j, sumq):
    rating_list = []
    sumk = []
    gids = []
    sum_query = """
        MATCH (s:Summary)
        RETURN s.content, s.gid
        """
    res = n4j.query(sum_query)
    for r in res:
        sumk.append(r['s.content'])
        gids.append(r['s.gid'])
    
    for sk in sumk:
        sk = sk[0]
        rate = call_llm(sys_p, "The two summaries for comparison are: \n Summary 1: " + sk + "\n Summary 2: " + sumq[0])
        if "totally not similar" in rate:
            rating_list.append(0)
        elif "not similar" in rate:
            rating_list.append(1)
        elif "general" in rate:
            rating_list.append(2)
        elif "very similar" in rate:
            rating_list.append(4)
        elif "similar" in rate:
            rating_list.append(3)
        else:
            print("llm returns no relevant rate")
            rating_list.append(-1)

    ind = find_index_of_largest(rating_list)
    # print('ind is', ind)

    gid = gids[ind]

    return gid


def quick_ret_top_k(n4j, sumq, k=5):
    """
    基于向量相似度检索top-k个最相似的文档子图
    
    Args:
        n4j: Neo4j数据库连接对象
        sumq: 查询摘要
        k: 返回结果数量
        
    Returns:
        list: 包含top-k个gid的列表
        list: 对应的相似度分数列表
    """
    sumk = []
    gids = []
    sum_query = """
        MATCH (s:Summary)
        RETURN s.gid, s.embedding
        """
    res = n4j.query(sum_query)
    for r in res:
        sumk.append(r['s.embedding'])
        gids.append(r['s.gid'])
    
    # 如果结果数量少于k，调整k值
    k = min(k, len(gids))
    
    # 将所有embedding转换为numpy数组
    embeddings_matrix = np.array(sumk)
    query_embedding = np.array(get_embedding(sumq[0]))
    
    # 计算相似度分数
    similarity_scores = np.dot(embeddings_matrix, query_embedding)
    similarity_scores = np.squeeze(similarity_scores) if similarity_scores.ndim == 2 else similarity_scores
    
    # 归一化分数
    if len(similarity_scores) > 0:
        similarity_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min() + 1e-10)
    
    # 获取top-k个索引
    if k == 1:
        top_indices = [np.argmax(similarity_scores)]
        top_scores = [similarity_scores[top_indices[0]]]
    else:
        # 获取排序后的索引
        top_indices = np.argsort(similarity_scores)[::-1][:k]
        top_scores = [similarity_scores[i] for i in top_indices]
    
    # 获取对应的gid
    top_gids = [gids[i] for i in top_indices]
    
    return top_gids, top_scores


# 示例用法
def retrieve_with_reranking(n4j, query, num_to_retrieve=5):
    """
    检索并重新排序结果
    
    Args:
        n4j: Neo4j数据库连接对象
        query: 查询文本
        num_to_retrieve: 需要检索的文档数量
        
    Returns:
        list: 检索到的文档内容
        list: 对应的分数
    """
    # 获取top-k个gid和分数
    top_gids, top_scores = quick_ret_top_k(n4j, [query], k=num_to_retrieve)
    
    # 如果没有找到结果，可以回退到其他检索方法
    if len(top_gids) == 0:
        print('没有找到匹配的摘要，回退到其他检索方法')
        # 这里可以调用其他检索方法
        return [], []
    
    # 根据gid获取对应的文档内容
    contents = []
    for gid in top_gids:
        # 查询该gid对应的文档内容
        content_query = """
            MATCH (n)
            WHERE n.gid = $gid AND NOT n:Summary
            RETURN n.content as content
            LIMIT 1
            """
        result = n4j.query(content_query, {"gid": gid})
        if result and 'content' in result[0]:
            contents.append(result[0]['content'])
        else:
            contents.append("未找到内容")
    
    return contents, top_scores
