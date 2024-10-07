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
    if args.dataset_name == 'jp':
        system_prompt = "あなたは人事分野の専門家です。"
    else:
        system_prompt = "You are an expert in the field of human resources."
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": df['prompt'][idx]},
    ]
    ans = tokenizer.apply_chat_template(conversation, tokenize=False)

    return ans


df = pd.read_csv(f'{args.input_dir}/task_candidate.csv', sep='\t').reset_index()
def gen(x):
    if args.dataset_name == 'jp':
        ans = f"100 個の具体的なタスクの説明と 1 つの具体的なポジションが与えられます。\n"
    else:
        ans = f"Now given 100 specific task descriptions and 1 specific position, \n"
    for i in range(100):
        ans += f"{i}." + x[f"task{i}"] + "\n"

    if args.dataset_name == 'jp':
        ans += f"対応する責任:\n{x['job_description']}\n"
        ans += f"上記の 100 個のタスクの説明から、職務責任に最も適合する 10 個のタスクの説明を選択します (タスクと責任の間の相関関係によって並べられ、相関関係が高いほど上位にランクされます)。\n回答には、次の形式を厳守してください: \n1.xxxx. (タスク x)\n2.xxxx. (タスク x)\n3.xxxx. (タスク x)\n"
    else:
        ans += f"corresponding responsibilities:\n{x['job_description']}\n"
        ans += "Please select the 10 task descriptions that best match the corresponding responsibilities of the position from the above 100 task descriptions (sorted from high to low in terms of degree of conformity). \nPlease strictly follow the following format for your answer: \n1.xxxx. (Task x)\n2.xxxx. (Task x)\n3.xxxx. (Task x)\n"
    return ans
df['prompt'] = df.apply(lambda x: gen(x), axis=1) 


length = len(df)
if args.dataset_name == 'jp':
    model_path = "../llms/Meta-Llama-3.1-8B-Instruct"
else:
    model_path = "../llms/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)   
prompts_all = [[idx,gen_prompt(idx)] for idx in tqdm(range(length))]

accelerator = Accelerator()
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
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)[0]
        result = tokenizer.decode(output_tokenized)
        results["outputs"].append(result)
        results["num_tokens"] += len(output_tokenized)
        with open(f'{args.output_dir}/{prompt[0]}.txt', 'w') as w:
            w.write(result)
    results=[ results ] 

results_gathered=gather_object(results)