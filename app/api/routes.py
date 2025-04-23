from fastapi import APIRouter, Depends, HTTPException
from api.models import (
    GraphCreationRequest, QueryRequest, RefLinkRequest, CSVImportRequest,
    GraphResponse, QueryResponse, RefLinkResponse, CSVImportResponse
)
from services.graph_service import GraphService
from services.query_service import QueryService
from core.database import get_neo4j_connection

router = APIRouter()

@router.post("/graph/build", response_model=GraphResponse)
async def build(request: GraphCreationRequest):
    """创建知识图谱"""
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

@router.post("/graph/build_doc")
async def build_doc():
    """创建知识图谱"""
    try:
        gs = GraphService()
        result = gs.build_docs_graph("/Users/hanhenry99/jianpei/medgraphragJP/data/dataset_json/medical_guidelines/diabetes/2型糖尿病分级诊疗与质量管理专家共识_1745206962.5506403_structured.json")
        return GraphResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建知识图谱失败: {str(e)}")

@router.post("/graph/qa", response_model=QueryResponse)
async def qa(request: QueryRequest):
    """查询知识图谱"""
    try:
        n4j = get_neo4j_connection()
        query_service = QueryService(n4j)
        # 不再需要传递 gid 参数，因为在 query_service.py 中会通过 seq_ret 获取
        result = query_service.query(request.query)
        return QueryResponse(**{"answer": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询知识图谱失败: {str(e)}")

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
    """从CSV导入知识图谱"""
    try:
        graph_service = GraphService()
        result = graph_service.import_from_csv(
            csv_path=request.csv_path,
            add_embeddings=request.add_embeddings
        )
        return CSVImportResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"从CSV导入知识图谱失败: {str(e)}")