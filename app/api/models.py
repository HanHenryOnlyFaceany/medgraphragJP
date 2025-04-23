from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# 请求模型
class GraphCreationRequest(BaseModel):
    content: str = Field(..., description="要处理的医疗文本内容")
    grained_chunk: bool = Field(True, description="是否使用细粒度分块")
    ingraphmerge: bool = Field(True, description="是否合并相似节点")

class QueryRequest(BaseModel):
    query: str = Field(..., description="查询问题")
    # gid 字段不再需要，因为在 query_service.py 中已经通过 seq_ret 获取

class RefLinkRequest(BaseModel):
    source_gid: str = Field(..., description="源图GID")
    target_gid: str = Field(..., description="目标图GID")

class CSVImportRequest(BaseModel):
    csv_path: str = Field(..., description="CSV文件路径")
    add_embeddings: bool = Field(True, description="是否添加嵌入向量")

# 响应模型
class GraphResponse(BaseModel):
    gid: str = Field(..., description="图元素组ID")
    # node_count: int = Field(..., description="节点数量")
    # relationship_count: int = Field(..., description="关系数量")
    message: str = Field("处理成功", description="处理结果消息")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="查询回答")
    # references: List[str] = Field([], description="引用的参考资料")

class RefLinkResponse(BaseModel):
    source_gid: str = Field(..., description="源图GID")
    target_gid: str = Field(..., description="目标图GID")
    link_count: int = Field(..., description="创建的引用关系数量")
    links: List[Dict[str, Any]] = Field([], description="创建的引用关系详情")

class CSVImportResponse(BaseModel):
    gid: str = Field(..., description="导入的图的组ID")
    node_count: int = Field(..., description="导入的节点数量")
    relationship_count: int = Field(..., description="导入的关系数量")