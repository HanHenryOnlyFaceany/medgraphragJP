import time
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import MistralConfig, OllamaConfig
# from camel.configs import PPIOConfig

from camel.loaders import UnstructuredIO
from camel.storages import Neo4jGraph
from camel.retrievers import AutoRetriever
from camel.embeddings import MistralEmbedding
from camel.types import StorageType
from camel.agents import ChatAgent, KnowledgeGraphAgent
from camel.messages import BaseMessage

import os
from getpass import getpass

# Prompt for the API key securely
# silicon_api_key = getpass('Enter your API key: ')

os.environ["PPIO_API_KEY"] = "sk_J7l6-K7MQB_9aPomZCuXWXrmxEUF_U91rXvGfRypmj0"
os.environ["MISTRAL_API_KEY"] = "xA4UrUKm5aON4AnU83GbfuVy12NtGlnB"

url=os.getenv("NEO4J_URL")
username=os.getenv("NEO4J_USERNAME")
password=os.getenv("NEO4J_PASSWORD")

# Set Neo4j instance
n4j = Neo4jGraph(
    url=url,
    username=username,             # Default username
    password=password     # Replace 'yourpassword' with your actual password
)

# # # Set up model
# PPIO_DP_V_3_turbo = ModelFactory.create(
#     model_platform=ModelPlatformType.PPIO,
#     model_type=ModelType.PPIO_QWEN_2_5_72B,
#     model_config_dict=PPIOConfig(temperature=0.2).as_dict(),
# )

# Set up model
mistral_large_2 = ModelFactory.create(
    model_platform=ModelPlatformType.MISTRAL,
    model_type=ModelType.MISTRAL_LARGE,
    model_config_dict=MistralConfig(temperature=0.2).as_dict()
)

# # You can also set up model locally by using ollama
# mistral_large_2_local = ModelFactory.create(
#     model_platform=ModelPlatformType.OLLAMA,
#     model_type="mistral-large",
#     model_config_dict=OllamaConfig(temperature=0.2).as_dict(),
# )

# Set instance
uio = UnstructuredIO()
kg_agent = KnowledgeGraphAgent(model=mistral_large_2)

# # Set example text input
# text_example = """
# 患者是一名76岁男性，在接受左股动脉旁路移植术后住院，随后出院至康复机构。在被发现收缩压在70多，17小时无尿后，他再次出现在医院。在康复机构放置的Foley导尿管产生了100cc的浑浊/棕色尿液。此时阴茎口也可能有脓性分泌物。到急诊科就诊时，患者无主观主诉。在急诊室，他被发现收缩压为85。他接受了6升静脉输液，并短暂地开始服用多巴胺，因为他的收缩压在80左右。既往病史：1；冠状动脉疾病伴弥漫性三支血管病变；右优势，状态后近端左旋支架与远端左旋闭塞；右冠状动脉支架置入后状态（无经皮冠状动脉介入治疗，99%左旋对角，80%左前降支近端小，或80%左前降支远端小）。2. 充血性心力衰竭（射血分数15% - 20%）。3. 2型糖尿病伴神经病变。4. 高血压。5. 憩室病（结肠镜检查发现）。6. 阿尔茨海默氏痴呆。7. 胃肠道出血史（服用依替巴肽期间）。8. 心脏危险因素（基线肌酐为1.4 - 1.6）。9. 高胆固醇血症。10. 伤口培养中耐甲氧西林金黄色葡萄球菌和假单胞菌生长的历史。11. 严重周围血管疾病;左股动脉旁路移植术。12. 慢性无法愈合的足部溃疡。13. 最近右脚蜂窝组织炎。
# """


# # Create an element from given text
# element_example = uio.create_element_from_text(
#     text=text_example, element_id="0"
# )

# # Let Knowledge Graph Agent extract node and relationship information
# ans_element = kg_agent.run(element_example, parse_graph_elements=False)
# print(ans_element)


# # Check graph element
# graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
# print(graph_elements)



# # Add the element to neo4j database
# n4j.add_graph_elements(graph_elements=[graph_elements])
query="这位出院至康复机构的患者是否患有冠状动脉疾病？"


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
The Retrieved Context is {kg_result}
"""

user_msg = BaseMessage.make_user_message(
    role_name="CAMEL User", content=user_prompt
)

# Get response
agent_response = camel_agent.step(user_msg)

print(agent_response.msg.content)
