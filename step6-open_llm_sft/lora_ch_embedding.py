from datasets import load_dataset
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM,RewardTrainer,RewardConfig
import pandas as pd
import re
import random
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM,TrainingArguments,DataCollatorForLanguageodeling,AutoModel,Aut
import argparse
import json
import math
import time
from datasets import load_from_disk
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from datasets import Dataset
from torch.nn.functional import softmax
from sklearn.metrics import ndcg_score
import torch,nn.functional as F
from sklearn.metrics import roc_auc_score
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import copy
from peft import LoraConfig,PeftModel,get_peft_model
import os
from modeling_baichuan import BaichuanForCausaLLM
import csv

os.environ['CUDA_LAUNCH_BL0CKING']='1'

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default='/group/ars-group-songsiyu01/LLM_cv2jd/data/data_List/data_LLM4Rs_groundtruth_2.parquet',type=str,help='')
    parser.add_argument('--model_name',default='baichuan',type=str,help='')
    parser.add_argument('--output_dir',default='/individuaL/fangchuyu/icde/chatgpt_query/prompt_output',type=str,help='')
    parser.add_argument('--output_test_dir',default='/individuaL/fangchuyu/icde/chatgpt_query/output/lorabai13embbank2',type=str,help='')
    parser.add_argument('--num_candidate',default=150,type=int,help='num_candidate')
    parser.add_argument('--num_train_epochs',default=8,type=int,help='')
    parser.add_argument('--lr',default=3e-5,type=float,help='')
    parser.add_argument('--per_device_train_batch_size',default=8,type=int,help='')
    parser.add_argument('--gradient_accumulation_steps',default=8,type=int,help='')
    parser.add_argument('--init',default=True,type=bool,help='')
    parser.add_argument('--freeze_LLM',default=True,type=bool,help='')
    parser.add_argument('--Trans',default=True,type=bool,help='')
    parser.add_argument('--local_rank',type=int,default=0,help='')
    parser.add_argument('--answer_length',type=int,default=100,help='')
    parser.add_argument('--data_begin',type=int,default=0,help='')
    parser.add_argument('--data_end',type=int,default=0,help='')
    parser.add_argument('--task_name',default="task",type=str,help='')
    parser.add_argument('--train_or_test',default="train",type=str,help='')
    parser.add_argument('--checkpoint',default="",type=str,help='')
    parser.add_argument('--prompt_init', default=False, type=bool, help='')
    return parser.parse_args()

class data_process():
    def __init__(self, data, tokenizer, bos='<s>', eos='</s>', batch_size=2, shuffle=False):
        self.candidate, self.answer, self.t = [], [], []

        for x in data:
            self.candidate.append(x['candidate_list'])
            self.answer.append(x['answerlist'])
            self.t.append(x['text'])

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0
        self.tokenizer = tokenizer

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1

        candidate = self.candidate[start:offset]
        candidate = torch.tensor([eval(x) for x in candidate])
        answer = self.answer[start:offset]

        text = self.t[start:offset]
        return candidate, answer, text

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MLP(nn.Module):
    def __init__(self, emsize, hidden_size=1024, num_layers=2):
        super(MLP, self).__init__()

        self.first_layer = nn.Linear(emsize * 2, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.init_weights()

    def init_weights(self):
        # initrange =0.1
        # self.first_layer,weight.data.uniform_(-initrange,initrange)
        # self.first_layer,bias.data.zero_()
        # self,last_layer.weight.data.uniform_(-initrange,initrange)
        # self.last_layer.bias.data.zero_()
        nn.init.xavier_normal_(self.first_layer.weight)
        nn.init.xavier_normal_(self.last_layer.weight)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, user, item):
        ui_cat = torch.cat([user, item], 1)
        hidden = torch.relu(self.first_layer(ui_cat))
        for layer in self.layers:
            hidden = torch.relu(layer(hidden))
        rating = torch.squeeze(torch.sigmoid(self.last_layer(hidden)))
        return rating

class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()

    def forward(self, user, item):
        rating = torch.sigmoid(torch.sum(user * item, 1))
        return rating

class PromptEncoder(torch, nn.Module):
    def __init__(self, ncandidate, args):
        super().__init__()
        self.emsize = 5120
        self.args = args
        with open('/individuaL/fangchuyu/icde/chatgpt_query/prompt_init_3.json', 'r') as f:
            prompt_init_dict = json.load(f)
        self.prompt_dict = prompt_init_dict[self.args.task_name][self.args.model_name]
        # 学映射表征
        self.cand_embeddings = nn.Embedding(ncandidate, self.emsize)
        # nn.init.xavier_normal_(self.cand_embeddings.weight)

        # self.lstm_head = torch.nn.LSTM(input_size=self.emsize,
        #                                hidden_size = self.emsize // 2,
        #                                num_layers = 2,
        #                                dropout = 0.0,
        #                                bidirectional = True,
        #                                batch_first = True)
        self.mlp_head = nn.Sequential(nn.Linear(self.emsize, self.emsize),
                                      nn.ReLU(),
                                      nn.Linear(self.emsize, self.emsize))

    def prompt_init_f(self, inputs):
        batch_elements = [self.prompt_dict[int(idx.item())] for idx in inputs]

        return torch.tensor(batch_elements).cuda()

    def forward(self, inputs):
        if self.args.init:
            input_embeds = self.prompt_init_f(inputs)
            input_embeds = input_embeds.to(torch.bfloat16)
        else:
            # 学映射表征
            input_embeds = self.cand_embeddings(inputs)
        output_embeds = self.mlp_head(input_embeds)
        return output_embeds

class soft_promptuning_LLM(nn.Module):
    def __init__(self, LLM_path, ncandidate, args, freeze=False, lora_config=None, **kwargs):
        super(soft_promptuning_LLM, self).__init__()
        self.llm_model = AutoModelForCausalLM.from_pretrained(LLM_path, trust_remote_code=True)
        self.llm_model = get_peft_model(self.llm_model, lora_config)
        self.embed_layer = self.llm_model.get_input_embeddings()
        self.prompt_encoder = PromptEncoder(ncandidate, args)

    def forward(self, candidates, pre_text, behind_text, mask, labels, candidate_num, evaluate=False):
        if not evaluate:
            w_pre = self.embed_layer(pre_text)

            embedding_list = [w_pre]
            for i in range(candidate_num):
                embedding_list.append(self.prompt_encoder(candidates[:, i]).unsqueeze(1))
            w_beh = self.embed_layer(behind_text)
            embedding_list.append(w_beh)
            src = torch.cat(embedding_list, 1)
            output = self.llm_model(inputs_embeds=src, attention_mask=mask, labels=labels)
        else:
            w_pre = self.embed_layer(pre_text)
            embedding_list = [w_pre]
            cand_embed = self.prompt_encoder(candidates).unsqueeze(0)
            embedding_list.append(cand_embed)
            w_beh = self.embed_layer(behind_text)
            embedding_list.append(w_beh)
            src = torch.cat(embedding_list, 1)
            output = self.llm_model(inputs_embeds=src)
        return output

def train(data, model, tokenizer, args):
    model.train()

    text_loss = 0
    total_sample = 0
    while True:
        candidate, answer, text = data.next_batch()
        batch_size = len(answer)
        if args.task_name == 'bank':
            prefix_prompt = ['### Query: Now gives 77 descriptions of specific intents and 1 specific text,\n specific class: '] * batch_size
            behind_prompt = [f'specific text:{text[i]} \n Please sort the above 77 types of intents according to the degree of matching with the intent of the text.\n Please strictly follow the format of the answer given:\n1.xxxx\n2.xxxx\n3.'
                             f'xxxx\n.  ### Answer:{answer[i]}' for i in range(batch_size)]
        elif args.task_name == 'clinc':
            prefix_prompt = ['### Query: Now gives 150 descriptions of specific intents and 1 specific text,\n specific class: '] * batch_size
            behind_prompt = [f'specific text:{text[i]} \n Please sort the above 150 types of intents according to the degree of matching with the intent of the text.\n please strictly follow the format of the answer given:\n1.xxxx\n2.xxxx\n3.'
                             f'xxxx\n.  ### Answer:{answer[i]}' for i in range(batch_size)]
        else:
            prefix_prompt = ['###Query:现在给出100个具体任务的描述和1个具体的职位,\n具体任务:'] * batch_size
            behind_prompt = [f'对应的职责:{text[i]}\n请从上述100个任务描述中选择最符合该职位对应职责的10个任务描述(符合程度由高到底排序)。\n给出的答案请严格遵循下面的格式:\n1.xxxx。(任务x)\n2.xxxx。(任务x)\n3.xxxx。(任务x)\n……###Answer:{answer[i]}' for i in range(batch_size)]

        prefix_encoded_inputs = tokenizer(prefix_prompt, max_length=1024, truncation=True, padding=True, return_tensors='pt')
        prefix_seq = prefix_encoded_inputs['input_ids']
        prefix_mask = prefix_encoded_inputs['attention_mask']
        behind_encoded_inputs = tokenizer(behind_prompt, max_length=1024, truncation=False, padding=True, return_tensors='pt')
        behind_seq = behind_encoded_inputs['input_ids']
        behind_mask = behind_encoded_inputs['attention_mask']
        # answer_encoded_inputs = tokenizer(answer,max_length=1024,truncation=Trye, return_tensors='pt')
        # ahswer_seq= answer_encoded_inputs['input_ids']
        # answer_mask= answer_encoded_inputs['attention_mask']
        pad_lef = torch.ones(batch_size, args.num_candidate)
        pad_input = torch.cat([prefix_mask, pad_lef, behind_mask], 1)
        pred_left = torch.full((batch_size, args.num_candidate), -100)
        pred_prefix = torch.where(prefix_mask == 1, prefix_seq, torch.tensor(-100))
        pred_behind = torch.where(behind_mask == 1, behind_seq, torch.tensor(-100))
        # pred_answer=torch.where(answer_mask==1,answer_seq,torch.tensor(-100))
        prediction = torch.cat([pred_prefix, pred_left, pred_behind], 1)
        outputs = model(candidate.cuda(), prefix_seq.cuda(), behind_seq.cuda(), pad_input.cuda(), prediction.cuda(), args.num_candidate)
        loss = outputs.loss
        model.backward(loss)
        model.step()
        text_loss += batch_size * loss.item()
        total_sample += batch_size
        if data.step % 100 == 0 or data.step == data.total_step:
            cur_t_loss = text_loss / total_sample
            print('text ppl {:4.4f}I{:5d}/{:5d} batches'.format(math.exp(cur_t_loss), data.step, data.total_step))
            text_loss = 0
            total_sample = 0
        if data.step == data.total_step:
            break

def postprocessing(string):
    string = re.sub('\'s', '\'s', string)
    string = re.sub('\'m', '\'m', string)
    string = re.sub('\'ve', '\'ve', string)
    string = re.sub('n\'t', 'n\'t', string)
    string = re.sub('\'re', '\'re', string)
    string = re.sub('\'d', '\'d', string)
    string = re.sub('\'ll', '\'ll', string)
    string = re.sub('\(', '(', string)
    string = re.sub('\)', ')', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string

def ids2token(ids, tokenizer, eos):
    text = tokenizer.decode(ids)
    # text= postprocessing(text)
    text_1 = text.split(eos)[0]
    # tokens=[]
    # for token in text.split():
    #     if token == eos:
    #         break
    #     tokens.append(token)
    return text_1

def generate(data, model, tokenizer, args):
    model.eval()

    idss_predict = []
    rating_prediction = []
    if not os.path.exists(args.output_test_dir):
        os.makedirs(args.output_test_dir)
    test_f = open(os.path.join(args.output_test_dir, str(args.data_begin) + '.csv'), 'a')
    test_cf = csv.writer(test_f)
    columns_name = ['query', 'answer', 'response_sft']
    test_cf.writerow(columns_name)
    # for i in range(args.data_begin,args.data_end):
    #     message = result['query'][i]
    #     inputs = tokenizer.encode(message, return_tensors="pt").to('cuda')
    #     outputs = lora_model.generate(input_ids=inputs, max_new_tokens=512, top_p=0.8, temperature=0.95, do_sample=False, re
    #     out = tokenizer.decode(outputs[0])
    #     tmplist = [message, result['label'][i], result['top1'][i], result['top2'][i], result['top3'][i], result['top4'][i], result['top5'][i], out]
    #     test_cf.writerow(tmplist)

    with torch.no_grad():
        while True:
            candidate, answer, text = data.next_batch()
            batch_size = len(text)
            for i in range(batch_size):
                tmptext = text[i]
                if args.task_name == 'bank':
                    prefix_prompt = ['## Query:Now gives 77 descriptions of specific intents and 1 specific text,\n specific class:']
                    behind_prompt = [f'specific text:{tmptext} \n please sort the above 77 types of intents according to the degree of matching with the intent of the text.\n please strictly follow the format of the answer given:\n1.xxxx\n2.xxxx\n3.xxxx\n.  ### Answer:']
                elif args.task_name == 'clinc':
                    prefix_prompt = ['### Query: Now gives 150 descriptions of specific intents and 1 specific text,\n specific class: ']
                    behind_prompt = [f'specific text:{text[i]} \n please sort the above 150 types of intents according to the degree of matching with the intent of the text.\n please strictly follow the format of the answer given:\n1.xxxx\n2.xxxx\n3.xxxx\n.  ### Answer: ']
                else:
                    prefix_prompt = ['###Query:现在给出100个具体任务的描述和1个具体的职位,\n具体任务:']
                    behind_prompt = [f'对应的职责:{tmptext}\n请从上述100个任务描述中选择最符合该职位对应职责的10个任务描述(符合程度由高到低排序)。\n 给出的答案请严格遵循下面的格式:\n1.xxxx。(任务x)\n2.xxxx。(任务x)\n3.xxxx。(任务x)\n......###Answer:']
                prefix_encoded_inputs = tokenizer(prefix_prompt, max_length=1024, truncation=True, padding=True, return_tensors='pt')
                prefix_seq = prefix_encoded_inputs['input_ids'].cuda()
                prefix_mask = prefix_encoded_inputs['attention_mask']
                behind_encoded_inputs = tokenizer(behind_prompt, max_length=1024, truncation=False, padding=True, return_tensors='pt')
                behind_seq = behind_encoded_inputs['input_ids'].cuda()
                behind_mask = behind_encoded_inputs['attention_mask']

                for _ in range(args.answer_length):
                    outputs = model(candidate[i].cuda(), prefix_seq, behind_seq, None, None, args.num_candidate, evaluate=True)
                    last_token = outputs.logits[:, -1, :]
                    word_prob = torch.softmax(last_token, dim=-1)
                    token = torch.argmax(word_prob, dim=1, keepdim=True)
                    behind_seq = torch.cat([behind_seq, token], 1)

                decoded_encodings = tokenizer.decode(behind_seq[0])
                test_cf.writerow([tmptext, answer[i], decoded_encodings])
            if data.step == data.total_step:
                break

def main():
    # deepspeed.init_distributed()
    args = set_args()
    if args.model_name =='nanbeige':
        model_dir = '/data/.modelcache/common-crawl-data/model-repo/Nanbeige/Nanbeige-16B-Base-32K/def24a80e8d88dd098b011ec39e'
    elif args.model_name =='1lama':
        model_dir = '/data/.modelcache/common-crawl-data/model-repo/meta-1lama/Llama-2-13b-hf/d6110f793f3afcdd34fdd66f6d3df7e1'
    else:
        model_dir = '/data/.modelcache/common-crawl-data/model-repo/baichuan-inc/Baichuan2-13B-Base/52552291d4fdf6de0f1d138497'
    print('--------------------Loading data----------------------------')
    tokenizer= AutoTokenizer.from_pretrained(model_dir,bos_token='<s>',eos_token='</s>',pad_token='</s>',trust_remote_code=True)
    if args.task_name =='bank':
        args.num_candidate = 77
        with open('/individual/fangchuyu/task_classify/lotrdatasets/banking/id2label.json','r')as f:
            id2label=json.load(f)
        label2id={}
        for i in id2label:
            label2id[id2label[i]]=int(i)
        data= pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/llmdata/banktrain.csv')
        query = pd.read_csv('/individual/fangchuyu/task_classify/data/simitopdata/banking77.csv')
        text_list =list(query['text'])
        candidate_list= []
        answerlist =[]
        if args.data_end != args.data_begin:
            if args.data_end>len(data['text']):
                data = data.iloc[args.data_begin:].reset_index(drop=True)
            else:
                data = data.iloc[args.data_begin:args.data_end].reset_index(drop=True)
        for ind,i in enumerate(data['text']):
            index=text_list.index(i)
            text= query.iloc[index]['text']
            tmplist= []
            for i in range(1,78):
                tmplist.append(label2id[query.iloc[index][f"top{j}_name"].lower()])
            candidate_list.append(str(tmplist))
            tmpanswer = []
            for j in range(1, 6):
                if pd.isna(data[f'top{j}'][ind]): break
                tmpanswer.append(str(j) + '.' + data[f'top{j}'][ind])
            answerlist.append('\n'.join(tmpanswer))
    elif args.task_name == 'clinc':
        args.num_candidate = 150
        with open('/individual/fangchuyu/task_classify/lotrdatasets/clinc/id2label.json', 'r') as f:
            id2label = json.load(f)
        label2id = {}
        for i in id2label:
            label2id[id2label[i]] = int(i)
        data = pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/llmdata/clinctrain.csv')
        query = pd.read_csv('/individual/fangchuyu/task_classify/data/simitopdata/clinc150.csv')
        text_list = list(query['text'])
        candidate_list = []
        answerlist = []
        if args.data_end != args.data_begin:
            if args.data_end > len(data['text']):
                data = data.iloc[args.data_begin:].reset_index(drop=True)
            else:
                data = data.iloc[args.data_begin:args.data_end].reset_index(drop=True)
        for ind, i in enumerate(data['text']):
            index = text_list.index(i)
            text = query.iloc[index]['text']
            tmplist = []
            for i in range(1, 151):
                tmplist.append(label2id[query.iloc[index][f"top{j}_name"].lower()])
            candidate_list.append(str(tmplist))
            tmpanswer = []
            for j in range(1, 6):
                if pd.isna(data[f'top{j}'][ind]): break
                tmpanswer.append(str(j) + '.' + data[f'top{j}'][ind])
            answerlist.append('\n'.join(tmpanswer))
    elif args.task_name == 'task':
        args.num_candidate = 150
        with open('/individual/fangchuyu/task_classify/label.json', 'r') as f:
            id2label = json.load(f)
        label2id = {}
        for i in id2label:
            label2id[id2label[i]] = int(i)
        data = pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/llmdata/tasktrain.csv')
        query = pd.read_csv('/individual/fangchuyu/task_classify/data/simitopdata/task100.csv')
        text_list = list(query['text'])
        candidate_list = []
        answerlist = []
        if args.data_end != args.data_begin:
            if args.data_end > len(data['text']):
                data = data.iloc[args.data_begin:].reset_index(drop=True)
            else:
                data = data.iloc[args.data_begin:args.data_end].reset_index(drop=True)
        for ind, i in enumerate(data['text']):
            index = text_list.index(i)
            text = query.iloc[index]['text']
            tmplist = []
            for i in range(1, 101):
                tmplist.append(label2id[query.iloc[index][f"top{j}_name"].lower()])
            candidate_list.append(str(tmplist))
            tmpanswer = []
            for j in range(1, 6):
                if pd.isna(data[f'top{j}'][ind]): break
                tmpanswer.append(str(j) + '.' + data[f'top{j}'][ind])
            answerlist.append('\n'.join(tmpanswer))

    data['candidate_list'] = candidate_list
    data['answerlist'] = answerlist
    print('------------------Loading model-----------------------')
    ntoken = len(tokenizer)
    if args.model_name == 'baichuan':
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["W_pack"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    model= soft_promptuning_LLM(model_dir, args.num_candidate, args, freeze=args.freeze_LLM, lora_config=peft_config).half()
    model.llm_model.resize_token_embeddings(ntoken)
    if args.train_or_test == 'train':
        # 训练数据集构建
        data = data.sample(frac=1.0)
        data = Dataset.from_pandas(data)
        train_data = data_process(data, tokenizer)
        model_d, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config_params=
        {"bf16":{
        "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "1r": 3e-5,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_1r": 0,
                "warmup_max_1r": 3e-5,
                "warmup_num_steps": 500
            }
        },
        "zero_optimization":{
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter":True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },

        "steps_per_print": 2000,
        "train_micro_batch_size_per_gpu": 2,
        "wall_clock_breakdown": False
        })

        print('---------------- training prompt -------------------')
        best_val_loss = float('inf')
        endure_count = 0
        best_epoch = 0
        for epoch in range(1, args.num_train_epochs + 1):
            train(train_data, model_d, tokenizer, args)
            model_d.save_16bit_model(args.output_dir, 'sp_cf_t}.bin', format(epoch))
        print('-------------train finish!!!-------------')
        print('best_epoch {} | loss {}'.format(best_epoch, best_val_loss))
    else:
        print('----------------- infering ------------------')
        data = Dataset.from_pandas(data)
        eval_data = data_process(data, tokenizer)
        checkpoint = torch.load(args.checkpoint)
        model.to('cpu')
        model.load_state_dict(checkpoint)
        for param_name, param in model.named_parameters():
            param.data = param.data.to(torch.bfloat16)
        generate(eval_data, model.cuda(), tokenizer, args)

if __name__ == "__main__":
    main()




