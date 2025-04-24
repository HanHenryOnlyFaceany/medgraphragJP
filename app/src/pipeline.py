from typing import Literal
from models import *
from utils import *
from modules import *
from construct import *


class Pipeline:
    """
    信息抽取流程主控类。
    该类负责组织和调度各个智能体（如SchemaAgent、ExtractionAgent、ReflectionAgent）以及案例库（CaseRepositoryHandler），
    实现从输入数据到信息抽取、知识图谱构建、案例更新等一系列流程的自动化。
    """

    def __init__(self, llm: BaseEngine):
        """
        初始化Pipeline，注入大模型引擎llm，并初始化各个智能体和案例库处理器。
        """
        self.llm = llm
        self.case_repo = CaseRepositoryHandler(llm = llm)
        self.schema_agent = SchemaAgent(llm = llm)
        self.extraction_agent = ExtractionAgent(llm = llm, case_repo = self.case_repo)
        self.reflection_agent = ReflectionAgent(llm = llm, case_repo = self.case_repo)

    def __check_consistancy(self, llm, task, mode, update_case):
        """
        检查当前llm与任务类型、模式的兼容性。
        若为OneKE模型且任务类型不支持，则抛出异常或自动切换为quick模式并关闭案例更新。
        """
        if llm.name == "OneKE":
            if task == "Base" or task == "Triple":
                raise ValueError("The finetuned OneKE only supports quick extraction mode for NER, RE and EE Task.")
            else:
                mode = "quick"
                update_case = False
                print("The fine-tuned OneKE defaults to quick extraction mode without case update.")
                return mode, update_case
        return mode, update_case

    def __init_method(self, data: DataPoint, process_method2):
        """
        初始化流程方法顺序。
        根据任务类型和传入的process_method2，确定schema_agent、extraction_agent、reflection_agent的执行方法。
        """
        default_order = ["schema_agent", "extraction_agent", "reflection_agent"]
        if "schema_agent" not in process_method2:
            process_method2["schema_agent"] = "get_default_schema"
        if data.task != "Base":
            process_method2["schema_agent"] = "get_retrieved_schema"
        if "extraction_agent" not in process_method2:
            process_method2["extraction_agent"] = "extract_information_direct"
        sorted_process_method = {key: process_method2[key] for key in default_order if key in process_method2}
        return sorted_process_method

    def __init_data(self, data: DataPoint):
        """
        根据任务类型初始化DataPoint对象的instruction和output_schema字段。
        """
        if data.task == "NER":
            data.instruction = config['agent']['default_ner']
            data.output_schema = "EntityList"
        elif data.task == "RE":
            data.instruction = config['agent']['default_re']
            data.output_schema = "RelationList"
        elif data.task == "EE":
            data.instruction = config['agent']['default_ee']
            data.output_schema = "EventList"
        elif data.task == "Triple":
            data.instruction = config['agent']['default_triple']
            data.output_schema = "TripleList"
        return data

    # main entry
    def get_extract_result(self,
                           task: TaskType,
                           three_agents = {},
                           construct = {},
                           instruction: str = "",
                           text: str = "",
                           output_schema: str = "",
                           constraint: str = "",
                           use_file: bool = False,
                           file_path: str = "",
                           truth: str = "",
                           mode: str = "quick",
                           update_case: bool = False,
                           show_trajectory: bool = False,
                           isgui: bool = False,
                           iskg: bool = False,
                           ):
        """
        信息抽取主入口方法。
        主要流程：
        1. 检查模型与任务模式的兼容性
        2. 构造并初始化DataPoint对象
        3. 根据模式确定各智能体的处理方法
        4. 顺序调用schema_agent、extraction_agent、reflection_agent进行信息抽取
        5. 可选：打印schema、抽取轨迹、结果
        6. 可选：构建知识图谱（KG）
        7. 可选：更新案例库
        8. 返回抽取结果、轨迹、前端schema、前端结果
        """
        # for key, value in locals().items():
        #     print(f"{key}: {value}")

        # Check Consistancy
        mode, update_case = self.__check_consistancy(self.llm, task, mode, update_case)

        # Load Data
        data = DataPoint(task=task, instruction=instruction, text=text, output_schema=output_schema, constraint=constraint, use_file=use_file, file_path=file_path, truth=truth)
        data = self.__init_data(data)
        if mode in config['agent']['mode'].keys():
            process_method = config['agent']['mode'][mode].copy()
        else:
            process_method = mode

        if isgui and mode == "customized":
            process_method = three_agents
            print("Customized 3-Agents: ", three_agents)

        sorted_process_method = self.__init_method(data, process_method)
        print("Process Method: ", sorted_process_method)

        print_schema = False #
        frontend_schema = "" #
        frontend_res = "" #

        # Information Extract
        for agent_name, method_name in sorted_process_method.items():
            agent = getattr(self, agent_name, None)
            if not agent:
                raise AttributeError(f"{agent_name} does not exist.")
            method = getattr(agent, method_name, None)
            if not method:
                raise AttributeError(f"Method '{method_name}' not found in {agent_name}.")
            data = method(data)
            if not print_schema and data.print_schema: #
                print("Schema: \n", data.print_schema)
                frontend_schema = data.print_schema
                print_schema = True
        data = self.extraction_agent.summarize_answer(data)

        # show result
        if show_trajectory:
            print("Extraction Trajectory: \n", json.dumps(data.get_result_trajectory(), indent=2))
        extraction_result = json.dumps(data.pred, indent=2)
        print("Extraction Result: \n", extraction_result)

        # construct KG
        if iskg:
            myurl = construct['url']
            myusername = construct['username']
            mypassword = construct['password']
            print(f"Construct KG in your {construct['database']} now...")
            cypher_statements = generate_cypher_statements(extraction_result)
            execute_cypher_statements(uri=myurl, user=myusername, password=mypassword, cypher_statements=cypher_statements)

        frontend_res = data.pred #

        # Case Update
        if update_case:
            if (data.truth == ""):
                truth = input("Please enter the correct answer you prefer, or just press Enter to accept the current answer: ")
                if truth.strip() == "":
                    data.truth = data.pred
                else:
                    data.truth = extract_json_dict(truth)
            self.case_repo.update_case(data)

        # return result
        result = data.pred
        trajectory = data.get_result_trajectory()

        return result, trajectory, frontend_schema, frontend_res
