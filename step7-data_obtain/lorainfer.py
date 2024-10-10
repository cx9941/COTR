from peft import PeftConfig
import pandas as pd
import transformers as tr
import torch
import argparse
import json
from datasets import Dataset
import csv
from datetime import datetime
import os
from peft import LoraConfig,PeftModel,get_peft_model
from transformers import AutoTokenizer,AutoModelForCausalLM
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',default="/home/data/qinchuan/TMIS/COTR/llms/Meta-Llama-3.1-8B-Instruct",type=str,heLp='')
    parser.add_argument('--lora_model',default="/home/data/qinchuan/TMIS/paper_code/output/data_obtain/llama/checkpoint-147",type=str,help='')
    parser.add_argument('--output_dir',default='/home/data/qinchuan/TMIS/paper_code/output/data_obtain/llama',type=str,help='')
    parser.add_argument('--checkpoint',default='/individual/fangchuyu/icde/chatgpt_query/sft_modeL/lorananbeig16cinc2/sp_cf_4.bin',type=str,help='')
    parser.add_argument('--num_train_epochs',default=3,type=int,help='')
    parser.add_argument('--lr',default=5e-4,type=float,help='')
    parser.add_argument('--per_device_train_batch_size',default=1,type=int,help='')
    parser.add_argument('--gradient_accumulation_steps',default=8,type=int,help='')
    parser.add_argument('--local_rank',type=int,default=0,help='')
    parser.add_argument('--data_type',type=str,default='train',help='')
    parser.add_argument('--data_begin',type=int,default=0,help='')
    parser.add_argument('--data_end',type=int,default=0,help='')
    parser.add_argument('--task_name',default="bank",type=str,help=')
    # parser.add_argument('--lora_r',type=int,default=8,help=')
    parser.add_argument("--deepspeed",type=str,default='/individual/fangchuyu/llmsft/ds_zero2.json',help="path to deepsped config file.")
    return parser.parse_args()


def main():
    args = set_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if 'Baichuan' in args.model_dir:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["w_pack"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = LoraConfig(
            r = 16,
            lora_alpha = 32,
            lora_dropout = 0.05,
            target_modules = ["q_proj", "v_proj"],
            bias = "none",
            task_type = "CAUSAL_LM",
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # lora_model=PeftModel.from_pretrained(model,args.lora_model).half()
    lora_model = get_peft_model(model, peft_config).half()
    lora_model = lora_model.to('cuda')
    checkpoint = torch.load(args.checkpoint)
    lora_model.to('cpu')
    lora_model.load_state_dict(checkpoint)
    lora_model = lora_model.to('cuda')

    if args.task_name == 'bank':
        # if args,data_type =='train':
        #     result = pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/llmdata/bank/train.csv')
        # elif args.data_type =='dev':
        #     result = pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/llmdata/bank/dev.csv')
        # else:
        #     result = pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/llmdata/bank/test.csv')

        with open("/home/data/qinchuan/TMIS/paper_code/banking_data/rank.json", 'r') as f:
            result = json.load(f)
        # result = pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/llmdata/banktrain.csv')
        # query = pd.read_csv('/individual/fangchuyu/task_classify/data/simitopdata/banking77.csv')
        # text_list = list(query['text'])

        query_list = []
        for i in result:
            text = i['text']
            name = "".join([f"class{j}." + i[f"top{j}"] for j in range(1, 61)])
            q = f'Now gives 77 descriptions of specific intents and 1 specific text,\n specific class:{name}\n,specific text:{text}\n Please sort the above 77 types of intents according to the degree of matching with the intent of the text. Please strictly follow the format of the answer given:\n1.xxxx\n2.xxxx\n3.xxxx\n......'
            query_list.append(q)
        result['query'] = query_list
    elif args.task_name == 'clinc':
        # if args.data_type =='train':
        #     result = pd.read_csv('/individuaL/fangchuyu/icde/chatgpt_query/llmdata/clinc/train.csv')
        # elif args.data_type =='dev':
        #     result = pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/Llmdata/clinc/dev.csv')
        # else:
        #     result = pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/llmdata/clinc/test.csv')
        result = pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/llmdata/clinctrain.csv')
        query = pd.read_csv('/individual/fangchuyu/task_classify/data/simitopdata/clinc150.csv')
        text_list = list(query['text'])
        query_list = []
        for i in result['text']:
            index = text_list.index(i)
            text = query.iloc[index]['text']
            name = "".join([f"class{j}." + query.iloc[index][f"top{j}_name"] for j in range(1, 151)])
            q = f'Now gives 150 descriptions of specific intents and 1 specific text,\n specific class: {name}\n,specific text:{text} \n Please sort the above 150 types of intents according to the degree of matching with the intent of the text.\n Please strictly follow the format of the answer given:\n1.xxxx\n2.xxxx\n3.xxxx\n......'
            query_list.append(q)
        result['query'] = query_list

    train_dataset = Dataset.from_pandas(result)

    def encode(item):
        temp_data = {}

        text = "### Query:" + item['query'] + '\n### Answer:'
        for i in range(1, 6):
            if pd.isna(item['top' + str(i)]): continue
            text += str(i) + '.' + item['top'+str(i)]+'\n'
        temp_data['text_answer'] = text
        return temp_data

    train_dataset = train_dataset.map(encode, batched=False, num_proc=16)
    test_f = open(os.path.join(args.output_dir, str(args.data_begin) + '.csv'), 'a')
    test_cf = csv.writer(test_f)
    columns_name = ['query', 'label', 'top1', 'top2', 'top3', 'top4', 'top5', 'response_sft']
    test_cf.writerow(columns_name)
    if args.data_end != args.data_begin:
        for i in range(args.data_begin, args.data_end):
            message = "### Query:" + result['query'][i] + '\n### Answer:'
            inputs = tokenizer.encode(message, return_tensors="pt").to('cuda')
            outputs = lora_model.generate(input_ids=inputs, max_new_tokens=512, top_p=0.8, temperature=0.95, do_sample=False, repetition_penalty=1.05, num_return_sequences=1)
            out = tokenizer.decode(outputs[0])
            tmplist = [message, result['label'][i], result['top1'][i], result['top2'][i], result['top3'][i], result['top4'][i], result['top5'][i], out]
            test_cf.writerow(tmplist)
    else:
        for i in range(len(result['query'])):
            message="### Query:" + result['query'][i]+'\n### Answer:'
            inputs = tokenizer.encode(message, return_tensors="pt").to('cuda')
            outputs =lora_model.generate(input_ids= inputs, max_new_tokens =512, top_p= 0.8, temperature=0.95, do_sample=False, repetition_penalty=1.05, num_return_sequences=1)
            out = tokenizer.decode(outputs[0])
            tmplist =[message, result['label'][i], result['top1'][i], result['top2'][i], result['top3'][i], result['top4'][i], result['top5'][i], out]
            test_cf.writerow(tmplist)

if __name__ == "__main__":
    main()

