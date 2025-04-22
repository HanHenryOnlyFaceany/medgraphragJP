from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import MistralConfig, OllamaConfig
from camel.loaders import UnstructuredIO
from camel.storages import Neo4jGraph
from camel.retrievers import AutoRetriever
from camel.embeddings import MistralEmbedding
from camel.types import StorageType
from camel.agents import ChatAgent, KnowledgeGraphAgent
from camel.messages import BaseMessage
import os
from camel.loaders import UnstructuredIO

os.environ["MISTRAL_API_KEY"] = "wA9hinyONKwuJ0Op2a6UTwtrlS8HEsor"
# Set retriever
# camel_retriever = AutoRetriever(
#     vector_storage_local_path="local_data/embedding_storage",
#     storage_type=StorageType.QDRANT,
#     embedding_model=MistralEmbedding(),
# )

# # Set one user query
# query="这位出院至康复机构的患者是否患有冠状动脉疾病？"
# # Set one user query
query="冠状动脉疾病是什么？"


# # Get related content by using vector retriever
# vector_result = camel_retriever.run_vector_retriever(
#     query=query,
#     contents="https://zh.wikipedia.org/zh-hans/%E5%86%A0%E7%8B%80%E5%8B%95%E8%84%88%E7%96%BE%E7%97%85",
# )

# # Show the result from vector search
# print(vector_result)

uio = UnstructuredIO()
kg_agent = KnowledgeGraphAgent()

elements = uio.parse_file_or_url(
    input_path="https://baike.baidu.com/item/%E5%86%A0%E7%8A%B6%E5%8A%A8%E8%84%89%E7%96%BE%E7%97%85/1117596"
)
chunk_elements = uio.chunk_elements(
    chunk_type="chunk_by_title", elements=elements
)

graph_elements = []
for chunk in chunk_elements:
    graph_element = kg_agent.run(chunk, parse_graph_elements=True)
    n4j.add_graph_elements(graph_elements=[graph_element])
    graph_elements.append(graph_element)

# Create an element from user query
query_element = uio.create_element_from_text(
    text=query, element_id="1"
)

# Let Knowledge Graph Agent extract node and relationship information from the qyery
ans_element = kg_agent.run(query_element, parse_graph_elements=True)

# Match the entity got from query in the knowledge graph storage content
kg_result = []
for node in ans_element.nodes:
    n4j_query = f"""
MATCH (n {{id: '{node.id}'}})-[r]->(m)
RETURN 'Node ' + n.id + ' (label: ' + labels(n)[0] + ') has relationship ' + type(r) + ' with Node ' + m.id + ' (label: ' + labels(m)[0] + ')' AS Description
UNION
MATCH (n)<-[r]-(m {{id: '{node.id}'}})
RETURN 'Node ' + m.id + ' (label: ' + labels(m)[0] + ') has relationship ' + type(r) + ' with Node ' + n.id + ' (label: ' + labels(n)[0] + ')' AS Description
"""
    result = n4j.query(query=n4j_query)
    kg_result.extend(result)

kg_result = [item['Description'] for item in kg_result]

# Show the result from knowledge graph database
print(kg_result)

# combine result from vector search and knowledge graph entity search
comined_results = str(vector_result) + "\n".join(kg_result)

# Set agent
sys_msg = BaseMessage.make_assistant_message(
    role_name="CAMEL Agent",
    content="""You are a helpful assistant to answer question,
        I will give you the Original Query and Retrieved Context,
    answer the Original Query based on the Retrieved Context.""",
)

camel_agent = ChatAgent(system_message=sys_msg,
                        model=mistral_large_2)

# Pass the retrieved information to agent
user_prompt=f"""
The Original Query is {query}
The Retrieved Context is {comined_results}
"""

user_msg = BaseMessage.make_user_message(
    role_name="CAMEL User", content=user_prompt
)

# Get response
agent_response = camel_agent.step(user_msg)

print(agent_response.msg.content)