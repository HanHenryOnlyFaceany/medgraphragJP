
# 转换txt文件为csv文件
# import csv

# txt_path = "/Users/hanhenry99/jianpei/medgraphragJP/dataset_cn/HITCPubMedKg_2/HITCPubMed-KGv2_0.txt"
# csv_path = "/Users/hanhenry99/jianpei/medgraphragJP/dataset_cn/HITCPubMedKg_2/HITCPubMed-KGv2_0.csv"

# with open(txt_path, "r", encoding="utf-8") as fin, \
#      open(csv_path, "w", encoding="utf-8", newline='') as fout:
#     writer = csv.writer(fout)
#     # 写表头
#     writer.writerow(["head_entity", "head_type", "relation", "tail_entity", "tail_type"])
#     for line in fin:
#         parts = line.strip().split('\t')
#         if len(parts) != 3 or '@@' not in parts[0] or '@@' not in parts[2]:
#             continue
#         head, relation, tail = parts
#         try:
#             head_entity, head_type = head.split('@@')
#             tail_entity, tail_type = tail.split('@@')
#             writer.writerow([head_entity, head_type, relation, tail_entity, tail_type])
#         except Exception:
#             continue

# print("转换完成，结果已保存到:", csv_path)

import pandas as pd
# 读取CSV文件
csv_path = "/Users/hanhenry99/jianpei/medgraphragJP/dataset_cn/HITCPubMedKg_2/HITCPubMed-KGv2_0.csv"
df = pd.read_csv(csv_path)
# 查看所有head_type和tail_type
# print(df['head_type'].unique())
# print(df['tail_type'].unique())

print(df['relation'].unique())