import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import pandas as pd
from tqdm import tqdm
from config import args

def gen_prompt(text_list):
    ans = '\n'.join([f"{i}.{v}" for i,v in enumerate(text_list)])
    return ans

accelerator = Accelerator()

df = pd.read_parquet(f'{args.input_dir}/task_candidate.parquet')
length = len(df)
prompts_all = [[idx, f"Given the following candidate tasks:\n"+ gen_prompt(df['task_candidate'][idx]) + f"\nPlease choose the best task from the task_candidate for the job description\n{idx},{df['job description'][idx]}\nReturn the best task\n"] for idx in range(length)]

model_path = "../llms/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)   

accelerator.wait_for_everyone()
start=time.time()

with accelerator.split_between_processes(prompts_all) as prompts:
    results=dict(outputs=[], num_tokens=0)
    for prompt in tqdm(prompts):
        prompt_tokenized=tokenizer(prompt[1], return_tensors="pt").to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)[0]
        result = tokenizer.decode(output_tokenized)
        result = result[result.find("Return the best task\n"):]
        results["outputs"].append(result)
        results["num_tokens"] += len(output_tokenized)
        with open(f'{args.output_dir}/{prompt[0]}.txt', 'w') as w:
            w.write(result)
    results=[ results ] 

results_gathered=gather_object(results)