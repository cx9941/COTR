import os
import pandas as pd
from transformers import BertTokenizer, BertModel
import re
from tqdm import tqdm
tqdm.pandas()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='jp')
parser.add_argument('--device', default='cuda')
args = parser.parse_args()
from utils import Task_Extraction

task_extraction = Task_Extraction(args.dataset_name)

if not os.path.exists(f'outputs/{args.dataset_name}'):
    os.makedirs(f'outputs/{args.dataset_name}')

df = pd.read_csv(f'data/{args.dataset_name}/job_description.csv', sep='\t')

df['task'] = df['job_description'].progress_apply(task_extraction.extract_tasks_from_description)
df['len'] = df['task'].apply(len)
df = df.sort_values('len', ascending=False)

df.to_parquet(f'outputs/{args.dataset_name}/job_re_description.parquet', index=None)
df.to_excel(f'outputs/{args.dataset_name}/job_re_description.xlsx', index=None, engine='xlsxwriter')