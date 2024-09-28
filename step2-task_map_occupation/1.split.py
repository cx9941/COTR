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

def extract_sentences_with_committed(text):
    # 正则表达式模式：匹配以句号、感叹号或问号结尾的句子，句子中包含"committed to"
    pattern = r'([A-Z][^.!?]*\bcommitted to\b[^.!?]*[.!?])'
    sentences = re.findall(pattern, text, re.IGNORECASE)
    return sentences

if not os.path.exists(f'outputs/{args.dataset_name}'):
    os.makedirs(f'outputs/{args.dataset_name}')

df = pd.read_csv(f'../data/{args.dataset_name}/all_data.csv', sep='\t')
df['task'] = df['text'].progress_apply(extract_sentences_with_committed)
df_task = df[df['task'].apply(len)>0].explode('task')
df_task = df_task.drop_duplicates('task')
df_task.to_csv(f'../data/{args.dataset_name}/all_data_task.csv', index=None, sep='\t')
df_task