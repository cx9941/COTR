# %%
import pandas as pd
import os
from config import args

import re
def extract_occupation(content):
    pattern = r"(\d+),([^\n]+)"
    matches1 = re.findall(pattern, content)
    
    pattern = r"\|\s*(\d+)\s*\|\s*(.+?)\s*\|"
    matches2 = re.findall(pattern, content)

    matches = matches2 if len(matches2) > len(matches1) else matches1
    return matches

# 读取源文件
all_data_task_topk = pd.read_parquet(f'{args.input_dir}/task_candidate.parquet')
all_data_task_topk['index'] = all_data_task_topk['index'].apply(str)
task_occupation = pd.read_csv(f'../data/{args.dataset_name}/task_occupation.csv', sep='\t')

# 读取输出文件
content_list = []
for i in range(len(os.listdir(args.output_dir))):
    content = open(f"{args.output_dir}/{i}.txt", 'r').read()
    content_list.append(content)
ans = pd.DataFrame(content_list)

# 处理输出文件
ans[1] = ans[0].apply(extract_occupation)
final_ans = ans.explode(1)
final_ans['index'] = final_ans[1].apply(lambda x: x[0] if type(x) == type(()) else '')
final_ans['occupation'] = final_ans[1].apply(lambda x: x[1]  if type(x) == type(()) and len(x)>0 else '')
final_ans = final_ans[['index', 'occupation']]
final_ans = final_ans[final_ans['index'].isin(all_data_task_topk['index'].tolist()) & final_ans['occupation'].isin(task_occupation['occupation'].tolist())].drop_duplicates('index')
final_ans = pd.merge(all_data_task_topk, final_ans, on=['index'])
final_ans.to_parquet(f'{args.result_dir}/result{args.turn}.parquet')

# 保存未处理好的文件
left_ans = all_data_task_topk[~all_data_task_topk['job description'].isin(final_ans['job description'])]
left_ans.to_parquet(f'{args.next_input_dir}/task_candidate.parquet')