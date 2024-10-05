# %%
import pandas as pd
from config import args

# df = pd.read_csv(f'outputs/{args.dataset_name}/job_re_description.csv', sep='\t').groupby(['job_title', 'job_description']).agg(list).reset_index()
df = pd.read_parquet(f'outputs/{args.dataset_name}/job_re_description.parquet')
print('match jd num', len(df[df['len']>0]))
filter_df = df[df['len']>0]
print('num of filter job', len(filter_df))
print('num of extracted job task:', filter_df['len'].sum().item())
filter_df.to_excel(f'outputs/{args.dataset_name}/job_re_description_have_task.xlsx', index=None)
filter_df

# %%
import os
filter_df = filter_df[['job_title', 'task']]
filter_df = filter_df.explode('task')
filter_df['task'] = filter_df['task'].apply(lambda x: x.split('。'))
filter_df = filter_df.explode('task')
filter_df = filter_df.drop_duplicates('task')
filter_df

# %%
if 'jp' in args.dataset_name:
    filter_df['task_len'] = filter_df['task'].apply(len)
else:
    filter_df['task_len'] = filter_df['task'].apply(lambda x: len(x.split(' ')))
filter_df = filter_df.sort_values('task_len', ascending=False)
filter_df = filter_df[(filter_df['task_len']>5) & (filter_df['task_len']<100)]
filter_df

# %%
if not os.path.exists(f'results/{args.dataset_name}'):
    os.makedirs(f'results/{args.dataset_name}')
filter_df.to_csv(f'results/{args.dataset_name}/job_re_description.csv', index=None, sep='\t')
filter_df.to_excel(f'results/{args.dataset_name}/{args.dataset_name}-job_re_description.xlsx', index=None)
print('num of filer task', len(filter_df))


