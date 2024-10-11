from peft import PeftConfig,PeftModel
import pandas as pd
import transformers as tr
import torch
import argparse
import json
from datasets import Dataset
import csv
from datetime import datetime
import os
from tqdm import tqdm
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default='/home/data/qinchuan/TMIS/paper_code/task_data/eu/eu-gpt.csv',type=str,help='')
    parser.add_argument('--rank_path', default='/home/data/qinchuan/TMIS/paper_code/task_data/eu/rank_bert.json', type=str,
                        help='')
    parser.add_argument('--model_dir',default="/home/data/qinchuan/TMIS/paper_code/llm_models/baichuan-inc/Baichuan2-7B-Chat",type=str,help='')
    parser.add_argument('--adapter_dir',default="/home/data/qinchuan/TMIS/paper_code/output/data_obtain/baichuan/checkpoint-30",type=str,help='')
    # parser,add_argument('--rLhf_adapter_dir',default="/group/ars-group-songsiyu01/LLM_cv2jd/output_rLhf_baichuan7B",type=str,help=')
    parser.add_argument('--output_dir',default='/home/data/qinchuan/TMIS/paper_code/output/data_obtain/baichuan/infer_nosft',type=str,help='')
    parser.add_argument('--data_begin',type=int,default=0,help='')
    parser.add_argument('--data_end',type=int,default=101,help='')
    parser.add_argument('--lora_true', type=bool, default=True, help='')
    parser.add_argument('--device_id', type=int, default=0, help='')
    return parser.parse_args()

def main():
    args = set_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # with open('/individual/fangchuyu/task_classify/datasets/banking/id2label.json','r')as f:
    #     id2label=json.load(f)
    # with open(args.data_path,'r')as f:
    #     data_list=[json.loads(line)for line in f]
    #     for x in data_list:
    #         x['label']=id2label[x['_id']]
    # df_train= pd.DataFrame(data_list)

    # result = pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/llmdata/banktrain.csv')
    # query = pd.read_csv('/individual/fangchuyu/task_classify/data/simitopdata/banking77.csv')
    # text_list =list(query['text'])
    with open(args.rank_path, 'r') as f:
        data = json.load(f)
    result = pd.read_csv(args.data_path)

    query_list= []
    texts = []
    labels = []
    # toplabels = [[] for _ in range(10)]
    for index, i in enumerate(result['job_description']):
        # index= text_list.index(i)
        # text = query.iloc[index]['text']

        text = i
        name="".join([f"class{j}."+ data[index][f"top{j}"]for j in range(1,101)])
        q = f'### Query: Now given 100 specific task descriptions and 1 specific position, \nSpecific tasks: {name}\n corresponding responsibilities: {text}\nPlease select the 10 task descriptions that best match the corresponding responsibilities of the position from the above 100 task descriptions (sorted from high to low in terms of degree of conformity). \nPlease strictly follow the following format for your answer: \n1.xxxx.\n2.xxxx.\n3.xxxx.\n ### Answer:'
        query_list.append(q)
        texts.append(text)
        labels.append(result['task'][index])
        # for j in range(10):
        #     toplabels[j].append(i[f"top{j + 1}"])

    newresult = {'query':query_list, 'text':texts, 'label':labels}
    # for j in range(10):
    #     newresult[f'top{j+1}'] = toplabels[j]
    result = pd.DataFrame(newresult)
    print('-------Data Loading Finished-------')

    train_dataset = Dataset.from_pandas(result)
    def encode(item):
        temp_data={}
        text=item['query']
        temp_data['text_only'] = text
        return temp_data

    # train_dataset =train_dataset.map(encode,batched=False,num_proc=16)

    result['response_sft'] = None
    # data['response_rlhf']= None
    # data['response_prompt']= None
    # base_model_name ="/data/.modelcache/common-crawl-data/model-repo/baichuan-inc/baichuan-7B/7a69737f9595a449377807212a96d
    device = f'cuda:{args.device_id}'
    model = tr.AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True).half().to(device)
    if args.lora_true:
        model = PeftModel.from_pretrained(model,args.adapter_dir,trust_remote_code=True).half().to(device)
    tokenizer = tr.AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    test_f = open(os.path.join(args.output_dir, str(args.data_begin) + '.csv'), 'w')
    test_cf = csv.writer(test_f)
    columns_name = ['query', 'label', 'response_sft']
    # columns_name = ['query', 'label', 'top1', 'top2', 'top3', 'top4', 'top5', 'response_sft']
    test_cf.writerow(columns_name)
    args.data_end = len(result['query'])
    for i in tqdm(range(args.data_begin, args.data_end)):
        message = result['query'][i]
        inputs = tokenizer.encode(message, return_tensors="pt").to(device)
        outputs = model.generate(input_ids=inputs, max_new_tokens=128, top_p=0.8, temperature=0.95, do_sample=False, repetition_penalty=1.05,num_return_sequences=1)
        out = tokenizer.decode(outputs[0])
        # tmplist = [message, result['label'][i], result['top1'][i], result['top2'][i], result['top3'][i], result['top4'][i], result['top5'][i], out]
        tmplist = [message, result['label'][i], out]
        test_cf.writerow(tmplist)

if __name__ == "__main__":
    main()

