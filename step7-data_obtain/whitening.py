from asyncio.log import logger
from cgitb import enable
from transformers import AutoTokenizer,AutoModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from tqdm.auto import tqdm,trange
import argparse
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import roc_curve,auc,classification_report
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BertClsModel(nn.Module):
    def __init__(self,model_path='/home/data/qinchuan/TMIS/paper_code/llm_models/AI-ModelScope/bert-base-cased'):
        # def _init_(self,model_path='/code/fangchuyu/job_classify/boss_bert'):
        super(BertClsModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert_emb = AutoModel.from_pretrained(model_path,output_attentions=True)
        self.embed_layer = self.bert_emb.get_input_embeddings()

    def meanpooling(self,last_hidden_state,attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded,1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask=torch.clamp(sum_mask,min=1e-9)
        mean_embeddings = sum_embeddings/sum_mask
        return mean_embeddings

    def forward(self, ids, attentions):
        output = self.bert_emb(input_ids=ids.to(device),
                               attention_mask=attentions.to(device))
        hidden_states = self.meanpooling(output[0], attentions)
        return hidden_states

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="process a list of texts with BERT")
    parser.add_argument("--f",default='0',type=str)
    args = parser.parse_args()
    model_path='/home/data/qinchuan/TMIS/paper_code/llm_models/AI-ModelScope/bert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path,output_hidden_states=True,output_attentions=True).to(device)
    label_data = pd.read_csv('/home/data/qinchuan/TMIS/COTR/step3-bert_recall/data/eu/task.csv', sep='\t')
    labeldata = list(label_data['DWA Title'])

    task_label= list(labeldata)
    print(len(task_label))
    label_list_ids =[]
    label_list_attentions = []
    for text in task_label:
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,# 根据需要进行调整
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        label_list_ids.append(encoded_text['input_ids'])
        label_list_attentions.append(encoded_text['attention_mask'])
    label_list_ids = torch.cat(label_list_ids,dim=0)
    label_list_attentions = torch.cat(label_list_attentions,dim=0)
    labeldataset = TensorDataset(label_list_ids,label_list_attentions)
    labeldataloader = DataLoader(labeldataset,batch_size=1000,shuffle=False)
    label_hidden =[]
    with torch.no_grad():
        model.eval()
        for batch in labeldataloader:
            label_tensor_ids,label_tensor_attentions = batch
            outputs = model(label_tensor_ids.to(device),label_tensor_attentions.to(device))
            all_hidden_states = outputs.hidden_states
            label_hidden.append(torch.mean((all_hidden_states[1]+ all_hidden_states[-1])/ 2,dim=1))
    label_hidden = torch.cat(label_hidden)

    # data_frames1= pd.read_csv('/individual/fangchuyu/icde/chatgpt_query/llmdata/banktrain.csv')
    data_frames1 = pd.read_csv('/home/data/qinchuan/TMIS/paper_code/task_data/eu/eu-gpt.csv')
    tasks = data_frames1["job_description"]
    text_list = list(tasks)

    input_ids = []
    attention_masks = []
    for text in text_list:
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,  # 根据需要进行调整
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_text['input_ids'])
        attention_masks.append(encoded_text['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    dataset = TensorDataset(input_ids, attention_masks)
    batch_size = 8
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        model.eval()
        pred = []
        truth = []
        data_table = []
        text_hidden = []
        train_bar = tqdm(total=len(dataloader))
        for ind, batch in enumerate(dataloader):
            # bert预测
            batch_input_ids, batch_attention_masks = batch
            bz = batch_input_ids.size(0)
            outputs = model(batch_input_ids.to(device), batch_attention_masks.to(device))
            all_hidden_states = outputs.hidden_states
            text_hidden.append(torch.mean((all_hidden_states[1] + all_hidden_states[-1]) / 2, dim=1))
            train_bar.update(5)

    text_hidden = torch.cat(text_hidden, dim=0)


    def whitening_torch_final(embeddings):
        mu = torch.mean(embeddings, dim=0, keepdim=True)
        cov = torch.mm((embeddings - mu).t(), embeddings - mu)
        u, s, vt = torch.svd(cov)
        w = torch.mm(u, torch.diag(1 / torch.sqrt(s)))
        embeddings = torch.mm(embeddings - mu, w)
        return embeddings

    num_text = text_hidden.shape[0]
    embeddings = whitening_torch_final(torch.cat([text_hidden, label_hidden], dim=0))
    text_hidden = embeddings[:num_text, :]
    label_hidden = embeddings[num_text:, :]
    simi_result = [torch.cosine_similarity(text_hidden[i], label_hidden, dim=1, eps=1e-08).unsqueeze(0) for i in range(num_text)]
    simi_result = torch.cat(simi_result)
    num = 0
    for i in range(num_text):
        tmplabel = data_frames1["task"][i]
        values, indices = torch.topk(simi_result[i], 200)
        tmp = []
        not_match = True
        for j in range(200):
            tmp.append(list(task_label)[indices[j].item()])
            if tmplabel in tmp and not_match:
                num += 1
                not_match = False
    print(len(text_list))
    print(num)
    print(num / len(text_list))

