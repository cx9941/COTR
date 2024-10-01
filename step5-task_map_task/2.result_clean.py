# %%
import pandas as pd
import os
from config import args
import re
# 读取源文件
all_data_task_topk = pd.read_parquet(f'{args.input_dir}/task_candidate.parquet').reset_index()
# 读取输出文件
ans = {'index':[], 'task': []}
for i in os.listdir(args.output_dir):
    idx = int(re.findall(r'\d+', i)[0])
    text = open(f"{args.output_dir}/{idx}.txt", 'r').read()
    content = text[text.find('Return the best task index'):]
    if len(re.findall(r'\d+', content)) < 1:
        continue
    choose_idx = int(re.findall(r'\d+', content)[0])
    candidates = all_data_task_topk['task_candidate'].tolist()[idx].tolist()
    if choose_idx < len(candidates):
        task = candidates[choose_idx]
    else:
        continue
    ans['task'].append(task)
    ans['index'].append(idx)
ans = pd.DataFrame(ans)

# 处理输出文件
final_ans = pd.merge(all_data_task_topk, ans, on=['index']).drop('index', axis=1)
final_ans = final_ans[~final_ans['task'].isna()]
final_ans.to_parquet(f'{args.result_dir}/result{args.turn}.parquet')

# 保存未处理好的文件
left_ans = all_data_task_topk[~all_data_task_topk['job description'].isin(final_ans['job description'])].drop('index', axis=1)
left_ans.to_parquet(f'{args.next_input_dir}/task_candidate.parquet', index=None)