from fastapi import APIRouter, Depends, HTTPException
from api.models import (
    GraphCreationRequest, GraphJsonRequest, QueryRequest, RefLinkRequest, CSVImportRequest,
    GraphResponse, QueryResponse, RefLinkResponse, CSVImportResponse, Md2JsonRequest
)
from services.graph_service import GraphService
from services.query_service import QueryService
from core.database import get_neo4j_connection
from src.utils.util import *
from src.utils.md2json import *



router = APIRouter()

@router.post("/graph/build", response_model=GraphResponse)
async def build(request: GraphCreationRequest):
    """创建顶层知识图谱"""
    try:
        graph_service = GraphService()
        result = graph_service.create_graph(
            content=request.content,
            grained_chunk=request.grained_chunk,
            ingraphmerge=request.ingraphmerge
        )
        return GraphResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建知识图谱失败: {str(e)}")

@router.post("/graph/build-doc")
async def build_doc(request: GraphJsonRequest):
    """创建中层知识图谱"""
    try:
        gs = GraphService()
        result = gs.build_docs_graph(request.file_path)
        return GraphResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建知识图谱失败: {str(e)}")

@router.post("/graph/precise-qa", response_model=QueryResponse)
async def precise_qa(request: QueryRequest):
    """精准查询知识图谱（单个文档）"""
    try:
        n4j = get_neo4j_connection()
        query_service = QueryService(n4j)
        # 不再需要传递 gid 参数，因为在 query_service.py 中会通过 seq_ret 获取
        result = query_service.precise_query(request.query)
        return QueryResponse(**{"answer": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询知识图谱失败: {str(e)}")


@router.post("/graph/qucik-qa", response_model=QueryResponse)
async def quick_qa(request: QueryRequest):
    """快速查询知识图谱（多个文档）"""
    try:

        n4j = get_neo4j_connection()
        query_service = QueryService(n4j)
        # 生成查询嵌入向量
        query_embeddings = query_service.get_query_embedding(request.query)
        # 不再需要传递 gid 参数，因为在 query_service.py 中会通过 seq_ret 获取

        result, meta_doc, reference_triples = query_service.quick_query(request.query, query_embeddings, k=5)
        return QueryResponse(**{
            "answer": result,
            "meta_doc": meta_doc,
            "reference_triples": reference_triples
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询知识图谱失败: {str(e)}")

@router.post("/graph/de-qucik-qa", response_model=QueryResponse)
async def de_quick_qa(request: QueryRequest):
    """快速查询知识图谱（多个文档）"""
    try:

        n4j = get_neo4j_connection()
        query_service = QueryService(n4j)
        sentences = split_sentences(request.query)
        # 生成查询嵌入向量
        query_embeddings = query_service.get_querys_embedding(sentences)
        # 不再需要传递 gid 参数，因为在 query_service.py 中会通过 seq_ret 获取

        result, meta_doc, reference_triples = query_service.quick_query(request.query, query_embeddings, k=5)
        return QueryResponse(**{
            "answer": result,
            "meta_doc": meta_doc,
            "reference_triples": reference_triples
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询知识图谱失败: {str(e)}")

@router.post("/graph/edge-embedding")
async def edge_embedding(gid: str):
    """计算三元组的嵌入向量，存在在边属性"""
    try:
        n4j = get_neo4j_connection()
        edges_embedding(n4j, gid)
        return {"message:计算边的嵌入向量成功！" }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计算边的嵌入向量失败: {str(e)}")

@router.post("/graph/sum-embedding")
async def sum_embedding(gid: str):
    """计算医学结构化总结的嵌入向量"""
    try:
        n4j = get_neo4j_connection()
        sums_embedding(n4j, gid)
        return {"message:计算边的嵌入向量成功！" }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计算边的嵌入向量失败: {str(e)}")


@router.post("/graph/reference", response_model=RefLinkResponse)
async def create_reference_links(request: RefLinkRequest):
    """创建图之间的引用关系"""
    try:
        graph_service = GraphService()
        result = graph_service.create_reference_links(
            source_gid=request.source_gid,
            target_gid=request.target_gid
        )
        return RefLinkResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建引用关系失败: {str(e)}")

@router.post("/graph/import-csv", response_model=CSVImportResponse)
async def import_from_csv(request: CSVImportRequest):
    """从CSV导入知识图谱，构建底层知识图谱"""
    try:
        graph_service = GraphService()
        result = graph_service.import_from_csv(
            csv_path=request.csv_path,
            add_embeddings=request.add_embeddings
        )
        return CSVImportResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"从CSV导入知识图谱失败: {str(e)}")


@router.post("/graph/md2json")
async def md2json(request: Md2JsonRequest):
    """md转Json医学文档结构化接口"""
    try:
        input_md = request.file_path
        output_json = input_md.replace('.md', '_structured.json')
        process_md_to_json_with_chunks(input_md, output_json, max_paragraphs=request.max_paragraphs)
        return {"message": "结构化处理完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"md转Json医学文档结构化接口失败: {str(e)}")

@router.post("/graph/delete")
async def delete_graph(gid: str):
    """删除指定gid的文档及其相关节点"""
    try:
        graph_service = GraphService()
        result = graph_service.delete_graph_by_gid(gid)
        return {"message": f"成功删除gid为{gid}的文档"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")