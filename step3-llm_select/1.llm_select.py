import os
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import pandas as pd
from tqdm import tqdm
from config import args

def gen_prompt(idx):
    text_list = df['task_candidate'][idx]
    prompt = '\n'.join([f"{i}.{v}" for i,v in enumerate(text_list)])
    ans = f"Given the job description:\n\n{df['job description'][idx]}\n\nPlease choose the best task from the task_candidate for the job description from the candidate tasks:\n\n{prompt}\nReturn the best task index\n"
    
    system_prompt = "You are an expert in the field of human resources."
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ans},
    ]
    ans = tokenizer.apply_chat_template(conversation, tokenize=False)

    return ans

accelerator = Accelerator()

df = pd.read_parquet(f'{args.input_dir}/task_candidate.parquet').reset_index()
length = len(df)
model_path = "../llms/Meta-Llama-3.1-8B-Instruct"
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
        if os.path.exists(f'{args.output_dir}/{prompt[0]}.txt'):
            continue
        prompt_tokenized=tokenizer(prompt[1], return_tensors="pt").to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)[0]
        result = tokenizer.decode(output_tokenized)
        results["outputs"].append(result)
        results["num_tokens"] += len(output_tokenized)
        with open(f'{args.output_dir}/{prompt[0]}.txt', 'w') as w:
            w.write(result)
    results=[ results ] 

results_gathered=gather_object(results)