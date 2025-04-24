# -*- coding: utf-8 -*-
import sys
sys.path.append("./app/src")
from models import *
from pipeline import *
import json
import nltk
nltk.download('punkt')

# model configuration
model = LocalServer(model_name_or_path="qwen2.5-32b-instruct")
pipeline = Pipeline(model)

# extraction configuration
Task = "Triple"
Text = """
肝硬化是由一种或多种病因引起的,以肝脏弥漫性纤维化、假小叶和再生结节形成为组织学特征的慢性进展性肝病。代偿期肝硬化无特殊临床表现，可以有门静脉高压和肝功能减退征象。如果出现腹水、食管胃底静脉曲张破裂出血、肝性脑病等并发症，则提示肝硬化进入失代偿状态。肝功能损伤包括轻度损伤、中度损伤和重度损伤3种级别。本共识应用Child-Pugh评分对肝硬化患者进行肝功能分级（表1)。Child-Pugh评分5\~6分为A级,其中5分为肝功能良好,6分为肝功能轻度损伤；7\~9分为B级，为肝功能中度损伤；10\~15分为C级，为肝功能重度损伤。
"""

Constraint = [["Disease", "Drug", "Food", "Check", "Department", "Producer", "Symptom", "Cure"],['belongs_to','common_drug',"do_eat","drugs_of","need_check","no_eat","recommand_drug","recommand_eat","has_symptom","acompany_with","cure_way"]]

# get extraction result
result, _, _, _ = pipeline.get_extract_result(task=Task, text=Text, constraint=Constraint, show_trajectory=False)
