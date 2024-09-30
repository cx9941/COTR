import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
from transformers import BertTokenizer, BertModel
import re
from tqdm import tqdm
tqdm.pandas()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='en')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()

action_verbs = ["Prepare", "Organize", "Design", "Lead", "Complete", "Participate"]
keywords = ["team", "software", "development", "feedback"]


def extract_job_tasks(job_description):
    tasks = []

    if action_verbs:
        verb_pattern = r"|".join(action_verbs)
        delimiters = r'[\.!?\*\-]'
        task_pattern = re.compile(rf'\b(?:{verb_pattern})\b.{{4,}}?{delimiters}', re.DOTALL)
        tasks.extend(re.findall(task_pattern, job_description))
    
    if keywords:
        # 使用关键词列表进行匹配，捕获包含这些关键词的完整句子
        keyword_pattern = r"|".join(keywords)
        task_pattern = re.compile(rf'\b.{{2,}}?(?:{keyword_pattern})\b.{{4,}}?{delimiters}', re.DOTALL)
        tasks.extend(re.findall(task_pattern, job_description))
    tasks = list(set(tasks))
    return tasks

if not os.path.exists(f'outputs/{args.dataset_name}'):
    os.makedirs(f'outputs/{args.dataset_name}')


df = pd.read_csv(f'../data/{args.dataset_name}/job_description.csv', sep='\t')
df['task'] = df['description'].progress_apply(extract_job_tasks)
df_task = df[df['task'].apply(len)>0].explode('task')
df_task = df_task.drop_duplicates('task')
df_task.to_csv(f'outputs/{args.dataset_name}/job_task.csv', index=None, sep='\t')
df_task