from transformers import AutoTokenizer,AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Subset,Dataset
import argparse
import json
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else"cpu")
batch_size =8
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TASKDATAJSON(Dataset):
    def __init__(self,datapath):
        super().__init__()
        self.desc = []
        self.label = []
        # self.position_code = []
        # self.position_name = []
        # self.jd_id = []
        # self.jd_title = []
        data = pd.read_csv(datapath)
        self.desc = list(data['job_description'])
        self.label = list(data['task'])
        # with open(datapath,'r')as file:
        #     for line in file:
        #         item = json.loads(line)
        #         self.desc.append(item['text'])
                # self.position_code.append(item['position_code'])
                # self.position_name.append(item['position_name'])
                # self.jd_id.append(item['job_title'])
                # self.jd_title.append(item['job_id'])

    def __len__(self):
        return len(self.desc)

    def __getitem__(self,index):
        # return self.desc[index],self.position_code[index],self.position_name[index],self.jd_id[index],self.jd_title[index]
        return self.desc[index], self.label[index]

    def collate(self):
        return lambda samples:map(list,zip(*samples))

class BertClsModel(nn.Module):
    def __init__(self,model_path="/home/data/qinchuan/TMIS/paper_code/llm_models/AI-ModelScope/bert-base-cased"):
        # def _init_(self,model_path='/code/fangchuyu/job_classify/boss_bert'):
        super(BertClsModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert_emb = AutoModel.from_pretrained(model_path,output_attentions=True)
        self.embed_layer = self.bert_emb.get_input_embeddings()
        self.template_num = 0
    def get_query(self,x_h,prompt_ge=True):
        "构建文本token序列"
        text_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_h))
        if len(text_tokens)>(512- self.template_num):
            text_tokens = text_tokens[:512 - self.template_num]
        return [text_tokens]

    def meanpooling(self,last_hidden_state,attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded,1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask=torch.clamp(sum_mask,min=1e-9)
        mean_embeddings = sum_embeddings/sum_mask
        return mean_embeddings

    def forward(self,desc):
        bz = len(desc)
        queries_list =[torch.LongTensor(self.get_query(desc[i])).squeeze(0)for i in range(bz)]
        queries = pad_sequence(queries_list,True,padding_value=0).long().to(device)
        attention_mask = queries != 0
        output = self.bert_emb(input_ids=queries.to(device), attention_mask=attention_mask.to(device))
        hidden_states = self.meanpooling(output.last_hidden_state,attention_mask)
        return hidden_states

def train_update(args,model,label_list):
    # 加载数据
    dataset = TASKDATAJSON(args.data_file)
    data_loader = DataLoader(dataset,batch_size=batch_size*2,collate_fn=dataset.collate())
    print(len(data_loader))
    output_path = args.output_file
    output_wf = open(output_path,'w')
    with torch.no_grad():
        model.eval()
        pred = []
        wdata = []
        label_hidden = model(label_list)
        num = 0
        allnum = 0
        # for texts, pos_id, pos_name, jd_id, jd_title in data_loader:
        for texts, labels in data_loader:
            bz = len(texts)
            task_hidden = model(texts)
            similarity = [torch.cosine_similarity(task_hidden[i].unsqueeze(0), label_hidden, dim=1, eps = 1e-08).unsqueeze(0) for i in range(bz)]
            similarity_raw = torch.cat(similarity)
            similarity = torch.softmax(similarity_raw, dim=-1).detach().cpu()
            pred_label = torch.argmax(similarity, dim=-1).detach().cpu()
            # for text, similist, predid, p_id, p_na, j_id, j_ti in zip(texts, similarity_raw, pred_label, pos_id, pos_name, jd_id, jd_title):
            for text, similist, predid, label in zip(texts, similarity_raw, pred_label, labels):
                allnum += 1
                tmp={'text':text}
                values, indices = torch.topk(similist, 200)
                not_match = True
                if label not in label_list:print('error')
                for x in range(200):
                    tmp[f'top{x + 1}'] = label_list[indices[x]]
                    # if label_list[indices[x]] == label and x < 10: not_match = False
                    if label_list[indices[x]] == label: not_match = False
                wdata.append(tmp)
                if not not_match: num += 1
        print(num / allnum)
        json.dump(wdata, output_wf)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity_thershold",default=0.9,type=float,help="which method to use")
    parser.add_argument("--label_file", default="/home/data/qinchuan/TMIS/COTR/step3-bert_recall/data/eu/task.csv", type=str,help="label")
    parser.add_argument("--data_file", default="/home/data/qinchuan/TMIS/paper_code/task_data/eu/eu-gpt.csv", type=str,help="data")
    parser.add_argument("--output_file", default="/home/data/qinchuan/TMIS/paper_code/task_data/eu/rank.json", type=str,help="output")
    args = parser.parse_args()

    model = BertClsModel().to(device)
    # with open(args.label_file, 'r')as f:
    #     label_list = json.load(f)
    label_data = pd.read_csv(args.label_file, sep='\t')
    label_list = list(label_data['DWA Title'])

    train_update(args, model, label_list)

if __name__ == '__main__':
    main()
