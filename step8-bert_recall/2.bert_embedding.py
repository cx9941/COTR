import torch
from transformers import BertTokenizer, BertModel
from model import BERTForTextMatching
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='en')
parser.add_argument('--file_name', type=str, default='candidate_jd')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--bert', type=str, default='raw')
args = parser.parse_args()

# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('../llms/bert-base-uncased')
if args.bert == 'raw':
    model = BertModel.from_pretrained('../llms/bert-base-uncased').to(args.device)
else:
    model = torch.load('../step3-bert_recall/CuBERT/outputs/en/2e-05_32_0.8/job_model.pt')

df = pd.read_csv(f'data/{args.dataset_name}/{args.file_name}.csv')  # 请将路径替换为你的实际路径
df['text'] = df['job_title'] + ':' + df['job_description']

texts = df['text'].tolist()  # 提取text列的内容

# 预处理函数，将文本转换为tokenizer格式
def preprocess_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = inputs['input_ids'].to(args.device)
    attention_mask = inputs['attention_mask'].to(args.device)
    if args.bert != 'raw':
        return input_ids, attention_mask
    else:
        return inputs

# 获取BERT的最后一层pooler output
def get_bert_embeddings(texts):
    if args.bert == 'raw':
        inputs  = preprocess_text(texts)
        with torch.no_grad():
            inputs = {i:v.to(args.device) for i,v in inputs.items()}
            outputs = model(**inputs).pooler_output
    else:
        input_ids,attention_mask  = preprocess_text(texts)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
    # pooler_output 是 [CLS] token 的embedding, 用于句子的表示
    return outputs

# 分批处理以避免显存不足，假设每批处理32条文本
batch_size = 32
embeddings = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = get_bert_embeddings(batch_texts)
    embeddings.append(batch_embeddings)

# 将所有批次的embedding拼接在一起
embeddings = torch.cat(embeddings, dim=0)

# 打印embedding的形状，通常为 (num_texts, 768)
print(embeddings.shape)

# 将BERT嵌入保存到文件
torch.save(embeddings, f'outputs/{args.dataset_name}_{args.file_name}_{args.bert}_embeddings.pt')