import pandas as pd
from utils import gen
from llm_func import get_gpt_response
from tqdm import tqdm
from config import args
import os
import time

final_ans = pd.read_csv(args.input_path)

if len(final_ans) > 0:
    final_ans['bert_task'] = final_ans['bert_task'].apply(eval)
    for i in range(100):
        final_ans[f"task{i}"] = final_ans['bert_task'].apply(lambda x: x[i])
    final_ans = final_ans.drop('bert_task', axis=1)
    final_ans['prompt'] = final_ans.apply(lambda x: gen(x, args.dataset_name), axis=1)
    for idx in tqdm(range(len(final_ans))):
        if os.path.exists(f'{args.output_dir}/{idx}.txt'):
            continue
        results = get_gpt_response(final_ans['prompt'][idx])
        with open(f'{args.output_dir}/{idx}.txt', 'w') as w:
            w.write(results)
        time.sleep(60)