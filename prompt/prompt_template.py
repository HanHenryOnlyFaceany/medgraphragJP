from langchain.prompts import PromptTemplate
from .prompt_example import *


# ==================================================================== #
#                           STRUCTURED AGENT                           #
# ==================================================================== #
STRUCTURED_INSTRUCTION = """
你是一名医学知识工程师。请对以下医学指南或论文的Markdown文本内容进行结构化分段和标签化处理，目的是为后续医学知识图谱的三元组抽取做准备。请遵循以下要求：

1. 仅保留与三元组抽取相关的中文内容，包括章节标题、中文摘要、中文关键词和正文段落，忽略图片、表格等无关内容。
2. 对每一部分内容加上结构化标签，标签类型包括：
   - SECTION_1, SECTION_2, ...（章节标题，数字代表层级）
   - ABSTRACT（中文摘要）
   - KEYWORDS（中文关键词）
   - PARAGRAPH（正文段落）
3. 对于每个标题下的正文内容，请按照换行符进行分段，每个段落单独保存为一个JSON对象，并在每个JSON对象中增加一个字段"title"，用于标明该段落所属的章节标题。
4. 每个结构化单元输出为一个JSON对象，包含"type"、"content"（或"title"）和"section_title"（所属标题）字段，最终以JSON数组形式输出。
5. 只保留包含中文字符的内容，去除所有英文内容和无关信息。
6. 输出示例：
[
{"type": "SECTION_1", "title": "2型糖尿病分级诊疗与质量管理专家共识"},
{"type": "ABSTRACT", "content": "【摘要】分级诊疗制度被认为是破解我国“看病难、看病贵”问题的治本之策...", "section_title": "2型糖尿病分级诊疗与质量管理专家共识"},
{"type": "KEYWORDS", "content": "【关键词】2型糖尿病；疾病管理；分级诊疗；医院，基层；全科医生；指南", "section_title": "2型糖尿病分级诊疗与质量管理专家共识"},
{"type": "SECTION_1", "title": "1分级诊疗的背景、原则、目标与依据"},
{"type": "PARAGRAPH", "content": "1.1分级诊疗的背景中国糖尿病患病人数估计达1.18亿，位列世界第一...", "section_title": "1分级诊疗的背景、原则、目标与依据"},
{"type": "PARAGRAPH", "content": "2018年，由深圳市医师协会内分泌代谢病科医师分会组织编写的《社区医生2型糖尿病管理流程与分级诊疗规范...", "section_title": "1分级诊疗的背景、原则、目标与依据"}
]

请严格按照上述要求处理以下Markdown文本内容，并输出结构化后的JSON数组。
"""



# ==================================================================== #
#                           SCHEMA AGENT                               #
# ==================================================================== #

# Get Text Analysis
TEXT_ANALYSIS_INSTRUCTION = """
**Instruction**: Please analyze and categorize the given text.
{examples}
**Text**: {text}

**Output Shema**: {schema}
"""

text_analysis_instruction = PromptTemplate(
    input_variables=["examples", "text", "schema"],
    template=TEXT_ANALYSIS_INSTRUCTION,
)

# Get Deduced Schema Json
DEDUCE_SCHEMA_JSON_INSTRUCTION = """
**Instruction**: Generate an output format that meets the requirements as described in the task. Pay attention to the following requirements:
    - Format: Return your responses in dictionary format as a JSON object.
    - Content: Do not include any actual data; all attributes values should be set to None.
    - Note: Attributes not mentioned in the task description should be ignored.
{examples}
**Task**: {instruction}

**Text**: {distilled_text}
{text}

Now please deduce the output schema in json format. All attributes values should be set to None.
**Output Schema**:
"""

deduced_schema_json_instruction = PromptTemplate(
    input_variables=["examples", "instruction", "distilled_text", "text", "schema"],
    template=DEDUCE_SCHEMA_JSON_INSTRUCTION,
)

# Get Deduced Schema Code
DEDUCE_SCHEMA_CODE_INSTRUCTION = """
**Instruction**: Based on the provided text and task description, Define the output schema in Python using Pydantic. Name the final extraction target class as 'ExtractionTarget'.
{examples}
**Task**: {instruction}

**Text**: {distilled_text}
{text}

Now please deduce the output schema. Ensure that the output code snippet is wrapped in '```',and can be directly parsed by the Python interpreter.
**Output Schema**: """
deduced_schema_code_instruction = PromptTemplate(
    input_variables=["examples", "instruction", "distilled_text", "text"],
    template=DEDUCE_SCHEMA_CODE_INSTRUCTION,
)


# ==================================================================== #
#                         EXTRACTION AGENT                             #
# ==================================================================== #

EXTRACT_INSTRUCTION = """
**Instruction**: You are an agent skilled in information extarction. {instruction}
{examples}
**Text**: {text}
{additional_info}
**Output Schema**: {schema}

Now please extract the corresponding information from the text. Ensure that the information you extract has a clear reference in the given text. Set any property not explicitly mentioned in the text to null.
"""

extract_instruction = PromptTemplate(
    input_variables=["instruction", "examples", "text", "schema", "additional_info"],
    template=EXTRACT_INSTRUCTION,
)

instruction_mapper = {
    'NER': "You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.",
    'RE': "You are an expert in relationship extraction. Please extract relationship triples that match the schema definition from the input. Return an empty list for relationships that do not exist. Please respond in the format of a JSON string.",
    'EE': "You are an expert in event extraction. Please extract events from the input that conform to the schema definition. Return an empty list for events that do not exist, and return NAN for arguments that do not exist. If an argument has multiple values, please return a list. Respond in the format of a JSON string.",
}

EXTRACT_INSTRUCTION_JSON = """
{{
    "instruction": {instruction},
    "schema": {constraint},
    "input": {input},
}}
"""

extract_instruction_json = PromptTemplate(
    input_variables=["instruction", "constraint", "input"],
    template=EXTRACT_INSTRUCTION_JSON,
)

SUMMARIZE_INSTRUCTION = """
**Instruction**: Below is a list of results obtained after segmenting and extracting information from a long article. Please consolidate all the answers to generate a final response.
{examples}
**Task**: {instruction}

**Result List**: {answer_list}

**Output Schema**: {schema}
Now summarize all the information from the Result List. Filter or merge the redundant information.
"""
summarize_instruction = PromptTemplate(
    input_variables=["instruction", "examples", "answer_list", "schema"],
    template=SUMMARIZE_INSTRUCTION,
)




# ==================================================================== #
#                          REFLECION AGENT                             #
# ==================================================================== #
REFLECT_INSTRUCTION = """**Instruction**: You are an agent skilled in reflection and optimization based on the original result. Refer to **Reflection Reference** to identify potential issues in the current extraction results.

**Reflection Reference**: {examples}

Now please review each element in the extraction result. Identify and improve any potential issues in the result based on the reflection. NOTE: If the original result is correct, no modifications are needed!

**Task**: {instruction}

**Text**: {text}

**Output Schema**: {schema}

**Original Result**: {result}

"""
reflect_instruction = PromptTemplate(
    input_variables=["instruction", "examples", "text", "schema", "result"],
    template=REFLECT_INSTRUCTION,
)

SUMMARIZE_INSTRUCTION = """
**Instruction**: Below is a list of results obtained after segmenting and extracting information from a long article. Please consolidate all the answers to generate a final response.

**Task**: {instruction}

**Result List**: {answer_list}
{additional_info}
**Output Schema**: {schema}
Now summarize the information from the Result List.
"""
summarize_instruction = PromptTemplate(
    input_variables=["instruction", "answer_list", "additional_info", "schema"],
    template=SUMMARIZE_INSTRUCTION,
)



# ==================================================================== #
#                            CASE REPOSITORY                           #
# ==================================================================== #

GOOD_CASE_ANALYSIS_INSTRUCTION = """
**Instruction**: Below is an information extraction task and its corresponding correct answer. Provide the reasoning steps that led to the correct answer, along with brief explanation of the answer. Your response should be brief and organized.

**Task**: {instruction}

**Text**: {text}
{additional_info}
**Correct Answer**: {result}

Now please generate the reasoning steps and breif analysis of the **Correct Answer** given above. DO NOT generate your own extraction result.
**Analysis**:
"""
good_case_analysis_instruction = PromptTemplate(
    input_variables=["instruction", "text", "result", "additional_info"],
    template=GOOD_CASE_ANALYSIS_INSTRUCTION,
)

BAD_CASE_REFLECTION_INSTRUCTION = """
**Instruction**: Based on the task description, compare the original answer with the correct one. Your output should be a brief reflection or concise summarized rules.

**Task**: {instruction}

**Text**: {text}
{additional_info}
**Original Answer**: {original_answer}

**Correct Answer**: {correct_answer}

Now please generate a brief and organized reflection. DO NOT generate your own extraction result.
**Reflection**:
"""

bad_case_reflection_instruction = PromptTemplate(
    input_variables=["instruction", "text", "original_answer", "correct_answer", "additional_info"],
    template=BAD_CASE_REFLECTION_INSTRUCTION,
)