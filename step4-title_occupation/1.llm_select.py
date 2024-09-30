import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import pandas as pd
from tqdm import tqdm
from config import args

def gen_prompt(idx):
    sub_df = df[idx*batch_size:(idx+1)*batch_size]
    ans = ['job title,occupation_candidate']
    for i,v in zip(sub_df['title'], sub_df["occupation_candidate"]):
        ans.append(f"{i},{v}")
    ans = '\n'.join(ans)
    ans = f"Given the following tables:\n {ans}\nPlease choose the best occupation from the occupation_candidate for each job title\n Return in the format of csv, the columns is ['job title', 'occupation']"

    system_prompt = "You are an expert in the field of human resources, Return in the format of csv."
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ans},
    ]
    ans = tokenizer.apply_chat_template(conversation, tokenize=False)
    return ans

accelerator = Accelerator()

df = pd.read_parquet(f'{args.input_dir}/occupation_candidate.parquet')
batch_size = 10
length = (len(df) - 1) // batch_size + 1

model_path = "../llms/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)   
prompts_all = [[idx, gen_prompt(idx)] for idx in range(length)]

model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
)
accelerator.wait_for_everyone()
start=time.time()

with accelerator.split_between_processes(prompts_all) as prompts:
    results=dict(outputs=[], num_tokens=0)
    for prompt in tqdm(prompts):
        prompt_tokenized=tokenizer(prompt[1], return_tensors="pt").to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)[0]
        result = tokenizer.decode(output_tokenized)
        results["outputs"].append(result)
        results["num_tokens"] += len(output_tokenized)
        with open(f'{args.output_dir}/{prompt[0]}.txt', 'w') as w:
            w.write(result)
    results=[ results ] 

results_gathered=gather_object(results)