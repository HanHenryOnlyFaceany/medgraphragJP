import time
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import MistralConfig, OllamaConfig
from camel.configs import PPIOConfig

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
# os.environ["MISTRAL_API_KEY"] = "xA4UrUKm5aON4AnU83GbfuVy12NtGlnB"

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
PPIO_DP_V_3_turbo = ModelFactory.create(
    model_platform=ModelPlatformType.PPIO,
    model_type=ModelType.PPIO_QWEN_2_5_72B,
    model_config_dict=PPIOConfig(temperature=0.2).as_dict(),
)

# Set up model
# mistral_large_2 = ModelFactory.create(
#     model_platform=ModelPlatformType.MISTRAL,
#     model_type=ModelType.MISTRAL_LARGE,
#     model_config_dict=MistralConfig(temperature=0.2).as_dict()
# )

# # You can also set up model locally by using ollama
# mistral_large_2_local = ModelFactory.create(
#     model_platform=ModelPlatformType.OLLAMA,
#     model_type="mistral-large",
#     model_config_dict=OllamaConfig(temperature=0.2).as_dict(),
# )

# Set instance
uio = UnstructuredIO()
kg_agent = KnowledgeGraphAgent(model=PPIO_DP_V_3_turbo)

# # Set example text input
# text_example = """
# 患者是一名76岁男性，名叫张三，社会病史：患者退休，入院前与妻子住在家中；入院前一天他一直住在这里。他是一个社交饮酒者，有40年的吸烟史；不过，他20年前就戒烟了。入院时体格检查：最初体格检查显示体温96.1华氏度，心率83，血压124/42（给予3升生理盐水），呼吸24,2升鼻插管时血氧饱和度100%。他的心跳和节奏都很正常。第一心音和第二心音正常。有2/6收缩期射血杂音，没有摩擦或跳动。他的双侧听诊无异常。他的腹部柔软，不触痛，不膨胀，并且有不活跃的肠音。他有明显的旁路移植脉搏，双侧足背肌和胫骨后肌脉搏；他的手术切口干净，干燥，完整。请注意，上述检查是由血管外科小组完成的，这是最初计划让病人住院的小组。相关的实验室资料：在最初的实验室评估中，患者的白细胞计数为12.7，血细胞比容为30.2，血小板为28.2万。pt为13.5，PTT为30.7，inr为1.3。
# 他的血清化学指标为钠136、钾5.4、氯99、碳酸氢盐25、尿素氮53、肌酐3.2、血糖91。他的钙含量为8.2，镁含量为2.4，磷酸盐含量为4.8。入院时还在进行血液培养，但最终呈阴性。入院时进行的尿液培养最初是待决的，但最终培养出了酵母。入院时进行的痰培养最初也是待定的，但最终也培养出了酵母。放射学/影像学：入院胸部x线片显示右肺动脉稳定突出；没有重点巩固领域；胸部整体稳定的外观与一项研究比较。没有充血性心力衰竭或肺炎的影像学证据。入院时心电图显示窦性心律，非特异性下/侧t波改变，肢体导联QRS电压低，v5和v6 t波变化 心电图日期。最初的腹部ct是有限的非对比检查，显示弥漫性血管钙化。无腹主动脉瘤或游离液体，冠状动脉钙化不完全成像，单纯性左肾囊肿，乙状结肠憩室病，前列腺肿大和部分钙化。医院系统课程：心血管：患者最初因低血压、尿量减少和急性肾功能衰竭入住血管重症监护病房；最有可能继发于假定的革兰氏阴性尿脓毒症（尽管从未有任何阳性培养数据证实这一诊断）。
# 入院当晚，在重症监护室登机时，患者突然出现心肺骤停。他被肾上腺素，利多卡因和四次直流电复律救活。他还插管以保护气道。采取这些措施后，患者恢复到窦性心律，收缩压约为100；从心跳停止到脉搏恢复的总时间约为16分钟。随后他需要双压维持血压。床边超声心动图显示轻微心包积液，左心室射血分数为20%至25%，为心动过速和右心室高动力；表明右侧填充压力升高。虽然这次骤停的确切病因尚不清楚，但最可能的触发因素是非q波心肌梗死，因为他的肌钙蛋白值在他骤停后升高到大于50。重复超声心动图显示轻度左心房扩张，射血分数15% - 20%，静息时区域壁运动异常，包括下、中、尖左心室运动，右心室收缩功能下降，中度二尖瓣反流。与前期研究的；左心室功能无变化。中度二尖瓣反流，右心室功能恶化。患者开始服用胺碘酮，随后开始口服胺碘酮。第二天，他开始静脉注射肝素，因为他的血清肌钙蛋白升高到大于50；持续用药72小时。鉴于他的肌钙蛋白升高和非q波心肌梗死，患者是心导管插入术的候选人。
# 然而，在医疗重症监护室小组与患者家属讨论后，决定不再采取进一步的侵入性手术，因为在上述复苏后，患者已被告知不要复苏/不要插管，其家属也不再希望采取积极的复苏措施。到目前为止，他停用了降压药，重新开始使用低剂量的受体阻滞剂和ace抑制剂。日晚，患者主诉胸骨下胸痛，经NPG sl和吗啡治疗后缓解。他也有v2和v3的st段下降，在疼痛缓解后恢复到基线。因此，患者再次开始使用肝素；虽然，当病人通过心脏酶排除心肌梗死时，这是停止的。他再次出现胸骨下胸痛；尽管如此，他的心电图没有变化，他再次排除了心肌梗死的可能性。入院第4天，患者开始出现充血性心力衰竭的迹象，给予积极的液体复苏，并开始用速尿进行轻度利尿。到2009年他转到普通医务室时，病人在重症监护室进行了积极的液体复苏后，仍然严重超载。因此，考虑到他有右侧充血性心力衰竭的体征和症状，他继续进行温和的利尿治疗。
# 出院康复时，患者已达到充分的利尿，2升鼻插管时氧饱和度大于95%。2. 感染性疾病/脓毒症：患者入院前因推测为革兰氏阴性尿脓毒症而服用左氧氟沙星、甲硝唑和万古霉素治疗右下肢蜂窝组织炎，并开始服用庆大霉素和哌拉西林/他唑巴坦，同时积极静脉输液补液。随后，他停用左氧氟沙星和甲硝唑，并开始使用氟康唑，因为他的尿液培养物中有酵母生长。他在医院的第三天停止使用庆大霉素，在医院的第五天停止使用氟康唑（根据传染病服务）。
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
query="张三出院后的情况怎么样？"


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
                        model=PPIO_DP_V_3_turbo)

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
