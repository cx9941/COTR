# %%
import pandas as pd
import os
from config import args

# 读取源文件
all_data_task_topk = pd.read_parquet(f'{args.input_dir}/task_candidate.parquet')
all_data_task_topk['index'] = all_data_task_topk['index'].apply(str)
task_list = set(all_data_task_topk.explode('task_candidate')['task_candidate'].tolist())
task_occupation = pd.read_csv(f'../data/{args.dataset_name}/task.csv', sep='\t')

# 读取输出文件
ans = {'index':[], 'task': []}
for i in os.listdir(args.output_dir):
    idx = re.findall(r'\d+', i)[0]
    content = open(f"{args.output_dir}/{idx}.txt", 'r').read()
    ans['task'].append(content)
    ans['index'].append(idx)
ans = pd.DataFrame(ans)

# 处理输出文件
final_ans = pd.merge(all_data_task_topk, ans, on=['index'])
final_ans['task'] = final_ans.apply(lambda x: [i for i in x['task_candidate'] if i in x['task']], axis=1)
final_ans = final_ans.explode('task')[['index', 'job description', 'task']]
final_ans.to_parquet(f'{args.result_dir}/result{args.turn}.parquet')

# 保存未处理好的文件
left_ans = all_data_task_topk[~all_data_task_topk['job description'].isin(final_ans['job description'])]
left_ans.to_parquet(f'{args.next_input_dir}/task_candidate.parquet')