import os
from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json
import pandas as pd
from tqdm import tqdm
from config import args
from utils import gen

def gen_prompt(idx):
    if args.dataset_name == 'jp':
        system_prompt = "あなたは人事分野の専門家です。"
    else:
        system_prompt = "You are an expert in the field of human resources."
    # conversation = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": df['prompt'][idx]},
    # ]
    # ans = tokenizer.apply_chat_template(conversation, tokenize=False)

    ans = df['prompt'][idx]

    return ans

df = pd.read_csv(args.input_path).reset_index()
if args.mode != 'all':
    df = df[df['tag']==1]
df['bert_task'] = df['bert_task'].apply(eval)

df['prompt'] = df.apply(lambda x: gen(x, args.dataset_name), axis=1) 


length = len(df)


tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)   
prompts_all = [[idx,gen_prompt(idx)] for idx in tqdm(range(length))]

accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.float16,
    trust_remote_code=True
)
accelerator.wait_for_everyone()
start=time.time()

with accelerator.split_between_processes(prompts_all) as prompts:
    results=dict(outputs=[], num_tokens=0)
    for prompt in tqdm(prompts):
        if os.path.exists(f'{args.output_dir}/{prompt[0]}.txt'):
            continue
        prompt_tokenized=tokenizer(prompt[1], return_tensors="pt").to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)[0]
        result = tokenizer.decode(output_tokenized)
        results["outputs"].append(result)
        results["num_tokens"] += len(output_tokenized)
        with open(f'{args.output_dir}/{prompt[0]}.txt', 'w') as w:
            w.write(result)
    results=[ results ] 

results_gathered=gather_object(results)