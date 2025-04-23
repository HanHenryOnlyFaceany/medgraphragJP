import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "医疗知识图谱API"
    
    # Neo4j配置
    NEO4J_URL: str = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # LLM配置
    PPIO_API_KEY: str = os.getenv("PPIO_API_KEY", "")
    PPIO_API_BASE: str = os.getenv("PPIO_API_BASE", "")
    
    class Config:
        case_sensitive = True

settings = Settings()