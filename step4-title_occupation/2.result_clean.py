# %%
import pandas as pd
import os
from config import args

import re
def extract_occupation(text):
    pattern = r'"([^"]+)","([^"]+)"'
    matches1 = re.findall(pattern, text)
    
    pattern = r'([^,]+),([^,]+)'
    matches2 = re.findall(pattern, text)

    matches = matches1 if len(matches1) > 4 else matches2
    return matches

# 读取源文件
occupation_candidate = pd.read_parquet(f'{args.input_dir}/occupation_candidate.parquet')
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
final_ans['title'] = final_ans[1].apply(lambda x: x[0] if type(x) == type(()) else '')
final_ans['occupation'] = final_ans[1].apply(lambda x: x[1]  if type(x) == type(()) and len(x)>0 else '')
final_ans = final_ans[['title', 'occupation']]
final_ans = final_ans[final_ans['title'].isin(occupation_candidate['title'].tolist()) & final_ans['occupation'].isin(task_occupation['occupation'].tolist())].drop_duplicates('title')
final_ans.to_parquet(f'{args.result_dir}/result{args.turn}.parquet')

# 保存未处理好的文件
left_ans = occupation_candidate[~occupation_candidate['title'].isin(final_ans['title'])]
left_ans.to_parquet(f'{args.next_input_dir}/occupation_candidate.parquet')