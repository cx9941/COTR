from datasets import load_dataset
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
import pandas as pd
from peft import LoraConfig, get_peft_model
import re
import random
from tqdm import tqdm
import os
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,DataCollatorForLanguageModeling,AutoModel
import argparse
import json
from tqdm import tqdm
import time
from datasets import load_from_disk
import pandas as pd
import os
from torch.utils.data import DataLoader
import re
import torch
import transformers
import random
import numpy as np
from datasets import Dataset
from torch.nn.functional import softmax
from sklearn.metrics import ndcg_score
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',default="baichuan",type=str,help='')
    parser.add_argument('--task_name',default="en",type=str,help='')
    parser.add_argument('--output_dir',default='temp_outputs',type=str,help='')
    parser.add_argument('--data_path', default='data/eu/eu-gpt.csv', type=str,
                        help='')
    parser.add_argument('--rank_path', default='data/eu/rank_bert.json',
                        type=str,
                        help='')
    parser.add_argument('--num_train_epochs',default=5,type=int,help='')
    parser.add_argument('--lr',default=5e-4,type=float,help='')
    parser.add_argument('--per_device_train_batch_size',default=1,type=int,help='')
    parser.add_argument('--gradient_accumulation_steps',default=4,type=int,help='')
    parser.add_argument('--local_rank',type=int,default=0,help='')
    # parser.add_argument('-lora_r',type=int,default=8,help='')
    parser.add_argument("--deepspeed",type=str,default='ds_zero2.json',help="path to deepspeed")
    return parser.parse_args()


def main():
    args = set_args()
    if os.path.exists(args.output_dir):
        return 0
    map_dir = {
        'llama': "../llms/Meta-Llama-3.1-8B",
        'baichuan': "../llms/Baichuan2-7B-Chat",
        'nanbeige': "../llms/Nanbeige2-8B-Chat",
    }
    args.model_dir = map_dir[args.model_name]
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    if 'Baichuan2' in args.model_dir:
        target_modules = ["W_pack"]
    else:
        target_modules = ["q_proj", "v_proj"]


    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # model = get_peft_model(model, peft_config)

    with open(args.rank_path, 'r') as f:
        data = json.load(f)
    # result = pd.read_csv('')
    # query = pd.read_csv('/individual/fangchuyu/task_classify/data/simitopdata/banking77.csv')

    # text_list =list(query['text'])
    # query_list= []
    # for i in result['text']:
    #     index=text_list.index(i)
    #     text = query.iloc[index]['text']
    #     name = "".join([f"class{j}." + query.iloc[index][f"top{j}_name"] for j in range(1, 78)])
    #     q = f'Now gives 77 descriptions of specific intents and 1 specific text,\n specific class:{name}\n,specific text:{text}\n please sort the above 77 types of intents according to the degree of matching with the intent of the text. \n Please strictly follow the format of the answer given:\n1.xxxx\n2.xxxx\n3.xxxx\n.'
    #     query_list.append(q)


    # with open("/home/data/qinchuan/TMIS/paper_code/banking_data/categories.json", 'r') as f:
    #     labellist = json.load(f)
    result = pd.read_csv(args.data_path)

    # candidate = labellist[:40]
    # tmpnum = {}
    # for i in candidate:
    #     tmpnum[i] = 0

    text_new = []
    label_new = []
    answerlist = []

    query_list= []
    for index in range(len(data)):
        text = data[index]['text']
        name = "".join([f"class{j}." + data[index][f"top{j}"] for j in range(1, 31)])

        if args.task_name == 'jp':
            q = f"### クエリ: 100 個の特定のタスクの説明と 1 つの特定のポジションが与えられます。\n具体的なタスク: {name}\n 対応する責任: {text}\n上記の 100 個のタスク記述から、そのポジションの対応する責任に最も一致する 10 個のタスク記述を選択してください (適合度の高い順に並べ替えられています)。\n回答では、次の形式に厳密に従ってください: \n1.xxxx.\n2.xxxx.\n3.xxxx.\n ### 回答:"
        else:
            q = f'### Query: Now given 100 specific task descriptions and 1 specific position, \nSpecific tasks: {name}\n corresponding responsibilities: {text}\nPlease select the 10 task descriptions that best match the corresponding responsibilities of the position from the above 100 task descriptions (sorted from high to low in terms of degree of conformity). \n Please strictly follow the following format for your answer: \n1.xxxx.\n2.xxxx.\n3.xxxx.\n ### Answer:'

        # if result['category'][index] in candidate and tmpnum[result['category'][index]] < 40:
        query_list.append(q)
        text_new.append(text)
        label_new.append(result['task'][index])
        answerlist.append(eval(result['gpt_task'][index]))

    newresult = {'query':query_list, 'text':text_new, 'category':label_new, 'answer':answerlist}
    result2 = pd.DataFrame(newresult)

    # result['query'] = query_list

    train_dataset = Dataset.from_pandas(result2)

    def encode(item):
        temp_data = {}

        text = "### Query:" + item['query'] + '\n### Answer:'

        for i in range(5):
            text += str(i + 1) + '.' + item['answer'][i] + '\n'


        # for i in range(1, 6):
        #     if pd.isna(item['top' + str(i)]): continue
        #     text += str(i) + '.' + item['top' + str(i)] + '\n'
        temp_data['text_answer'] = text
        return temp_data

    train_dataset = train_dataset.map(encode, batched=False, num_proc=16)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate = args.lr,
        weight_decay = 5e-4,
        adam_beta1 = 0.9,
        adam_beta2 = 0.95,
        num_train_epochs = args.num_train_epochs,
        # logging_dir=os.path.join(args.output_dir,'/logs'),
        fp16 = True,
        logging_strategy = "steps",
        logging_steps = 10,
        save_strategy = "epoch",
        report_to = 'none',
        deepspeed = args.deepspeed
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = train_dataset,
        dataset_text_field = "text_answer",
        max_seq_length = 2048,
        peft_config=peft_config,
    )

    trainer.train()

if __name__ == '__main__':
    main()


