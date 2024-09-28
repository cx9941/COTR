import os
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import pandas as pd
from tqdm import tqdm
from config import args


accelerator = Accelerator()

df = pd.read_parquet(f'{args.input_dir}/task_candidate.parquet')
batch_size = 10
length = (len(df) - 1) // batch_size + 1
prompts_all = [f"Given the following tables:\n {(df[idx*batch_size:(idx+1)*batch_size]).to_string()}\nPlease choose the best occupation from the occupation_candidate for each job description\n Return in the format of csv, the columns is ['index', 'occupation']" for idx in range(length)]

model_path = "../llms/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)   

accelerator.wait_for_everyone()
start=time.time()

with accelerator.split_between_processes(prompts_all) as prompts:
    results=dict(outputs=[], num_tokens=0)
    for prompt in tqdm(prompts):
        prompt_tokenized=tokenizer(prompt, return_tensors="pt").to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)[0]
        result = tokenizer.decode(output_tokenized)
        result = result[result.find("Return in the format of csv, the columns is ['index', 'occupation']"):]
        results["outputs"].append(result)
        results["num_tokens"] += len(output_tokenized)
    results=[ results ] 

results_gathered=gather_object(results)

cont = 0
for i,v in enumerate(results_gathered):
    for j in v['outputs']:
        with open(f'{args.output_dir}/{cont}.txt', 'w') as w:
            w.write(j)
        cont += 1