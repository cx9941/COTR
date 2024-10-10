# %%
import pandas as pd
import os
from config import args
import re
from tqdm import tqdm
all_data_task_topk = pd.read_csv(args.input_path).reset_index()
all_data_task_topk['bert_task'] = all_data_task_topk['bert_task'].apply(eval)
# 读取输出文件
final_ans = {'index':[], 'task': []}
for i in tqdm(os.listdir(args.output_dir)):
    idx = int(re.findall(r'\d+', i)[0])
    text = open(f"{args.output_dir}/{idx}.txt", 'r').read()
    content = text
    candidates = all_data_task_topk['bert_task'][idx]

    # ans = re.findall(r'\d\.+(.*?)\n', content)
    # results = []
    # # rgex = '|'.join([_ for _ in  candidates])
    # # results = re.findall(rf"{rgex}", content)
    # for line in content.split('\n'):
    #     for can in candidates:
    #         if can in line:
    #             results.append(can)
    #             break
    # if len(results) != 10:
    #     continue

    if args.dataset_name == 'jp':
        ans = re.findall(r'タスク\s*(\d+)', content)
    else:
        ans = re.findall(r'Task\s*(\d+)', content)
    
    ans = [int(ans[_]) for _ in range(len(ans)) if ans[_] not in ans[:_] and int(ans[_]) < 100]
    if len(ans) != 10:
        continue
    results = [candidates[i] for i in ans]

    final_ans['task'].append(results)
    final_ans['index'].append(idx)
final_ans = pd.DataFrame(final_ans)

# 处理输出文件
final_ans = pd.merge(all_data_task_topk[['index', 'job_title', 'job_description']], final_ans, on=['index']).drop('index', axis=1)
final_ans.to_csv(f'{args.result_dir}/result{args.turn}.csv', index=None)

# 保存未处理好的文件
left_ans = all_data_task_topk[~all_data_task_topk['job_description'].isin(final_ans['job_description'])].drop('index', axis=1)
left_ans.to_csv(f'{args.next_input_path}', index=None)