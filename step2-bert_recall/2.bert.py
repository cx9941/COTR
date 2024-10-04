import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='eu')
parser.add_argument('--device', default='cuda')
parser.add_argument('--data_type', default='title')
args = parser.parse_args()

if not os.path.exists(f'outputs/{args.dataset_name}'):
    os.makedirs(f'outputs/{args.dataset_name}')

df = pd.read_csv(f'data/{args.dataset_name}/{args.data_type}.csv', sep='\t')
text_list = df[args.data_type].tolist()
tokenizer = BertTokenizer.from_pretrained('../llms/bert-base-uncased')
model = BertModel.from_pretrained('../llms/bert-base-uncased').to(args.device)
def get_batch_embeddings(text_list, model, tokenizer, batch_size=512):
    embeddings = []
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch = text_list[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {i:v.to(args.device) for i,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu()
        embeddings.append(batch_embeddings)
    return torch.concat(embeddings)
text_embeddings = get_batch_embeddings(text_list, model, tokenizer, batch_size=512)
torch.save(text_embeddings, f'outputs/{args.dataset_name}/{args.data_type}_embedding.pt')