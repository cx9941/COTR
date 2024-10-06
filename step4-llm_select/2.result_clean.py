# %%
import pandas as pd
import os
from config import args
import re
# 读取源文件
all_data_task_topk = pd.read_csv(f'{args.input_dir}/task_candidate.csv', sep='\t').reset_index()
# 读取输出文件
final_ans = {'index':[], 'task': []}
for i in os.listdir(args.output_dir):
    idx = int(re.findall(r'\d+', i)[0])
    text = open(f"{args.output_dir}/{idx}.txt", 'r').read()
    content = text[text.find('<|start_header_id|>assistant<|end_header_id|>'):]
    ans = re.findall(r'\d\.+(.*?)\n', content)
    results = [re.sub(r'\d+\.', '', i).strip(' ') for i in ans]
    candidates = [all_data_task_topk[f'task{_}'][idx].strip(' ') for _ in range(100)]
    results = [_ for _ in results if _ in candidates]
    if len(results) != 10:
        continue
    final_ans['task'].append(results)
    final_ans['index'].append(idx)
final_ans = pd.DataFrame(final_ans)

# 处理输出文件
final_ans = pd.merge(all_data_task_topk[['index', 'job_title', 'job_description']], final_ans, on=['index']).drop('index', axis=1)
final_ans.to_parquet(f'{args.result_dir}/result{args.turn}.parquet')
final_ans.to_excel(f'{args.result_dir}/result{args.turn}.xlsx', index=None)

# 保存未处理好的文件
left_ans = all_data_task_topk[~all_data_task_topk['job_description'].isin(final_ans['job_description'])].drop('index', axis=1)
left_ans.to_csv(f'{args.next_input_dir}/task_candidate.csv', index=None, sep='\t')