import os
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import pandas as pd
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

def gen_prompt(idx):
    ans = f"给定工作描述:\n\n{df['job_description'][idx]}\n\n请判断这个描述是否属于一种任务\n返回“是”或“否”\n"
    
    system_prompt = "你是中文人力资源方面的专家，作出最准确的回答，回答是或者否."
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ans},
    ]
    ans = tokenizer.apply_chat_template(conversation, tokenize=False)

    return ans

accelerator = Accelerator()

df = pd.read_csv(f'data/jp-job_re_description_ch.csv', sep='\t')
length = len(df)
model_path = "../llms/Qwen2.5-7B-Instruct"
output_dir = f'outputs/seed{args.seed}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
tokenizer = AutoTokenizer.from_pretrained(model_path)   
prompts_all = [[idx,gen_prompt(idx)] for idx in tqdm(range(length))]


model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.float16,
)
accelerator.wait_for_everyone()
start=time.time()

with accelerator.split_between_processes(prompts_all) as prompts:
    results=dict(outputs=[], num_tokens=0)
    for prompt in tqdm(prompts):
        # if os.path.exists(f'{output_dir}/{prompt[0]}.txt'):
        #     continue
        prompt_tokenized=tokenizer(prompt[1], return_tensors="pt").to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)[0]
        result = tokenizer.decode(output_tokenized)
        results["outputs"].append(result)
        results["num_tokens"] += len(output_tokenized)
        with open(f'{output_dir}/{prompt[0]}.txt', 'w') as w:
            w.write(result)
    results=[ results ] 

results_gathered=gather_object(results)