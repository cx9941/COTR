# %%
import os
import pandas as pd
from config import args

df = pd.read_csv(f'../data/{args.dataset_name}/job_description.csv', sep='\t')
df = df[df['job_description'].apply(type)==type('')]
df = df[df['job_description'].apply(len)> 0]
df = df.drop_duplicates('job_description')
df.columns = ['job_title', 'job_description']
if not os.path.exists(f'data/{args.dataset_name}'):
    os.makedirs(f'data/{args.dataset_name}')
if args.dataset_name == 'en':
    df = df[df['job_description'].apply(lambda x: len(x.split(' '))<1000)]
df.to_csv(f'data/{args.dataset_name}/job_description.csv', index=None, sep='\t')
print('num of job description', len(df))
df