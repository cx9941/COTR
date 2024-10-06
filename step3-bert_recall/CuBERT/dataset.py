import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd

class TextMatchingDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, max_len):
        self.data = pd.read_csv(f'../data/{dataset_name}.csv', sep='\t')
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        job_text = f"{row['job_title']} - {row['job_description']}"
        task_text = f"{row['Element Name']} - {row['IWA Title']} - {row['DWA Title']}"

        job_encoding = self.tokenizer(
            job_text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt'
        )
        task_encoding = self.tokenizer(
            task_text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt'
        )

        return {
            'job_input_ids': job_encoding['input_ids'].flatten(),
            'job_attention_mask': job_encoding['attention_mask'].flatten(),
            'task_input_ids': task_encoding['input_ids'].flatten(),
            'task_attention_mask': task_encoding['attention_mask'].flatten()
        }
    
class TextDatasetDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, max_len):
        self.dataset_name = dataset_name
        self.data = pd.read_csv(f'../data/{dataset_name}.csv', sep='\t')
        if dataset_name == 'task':
            self.data['text'] = self.data['Element Name'] + '-' + self.data['IWA Title'] + '-' + self.data['DWA Title']
        else:
            self.data['text'] = self.data['job_title'] + '-' + self.data['job_description']

        if self.dataset_name == 'eval':
            task_data = pd.read_csv(f'../data/task.csv', sep='\t').reset_index()[['DWA Title', 'index']]
            self.data = pd.merge(self.data, task_data, on='DWA Title')

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        if self.dataset_name == 'eval':
            label = self.data.iloc[idx]['index']
        else:
            label = None
        text_encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        text_encoding = {i:v.flatten() for i,v in text_encoding.items()}
        
        if label == None:
            return text_encoding
        
        return text_encoding, label

def create_data_loader(tokenizer, max_len, batch_size, test_dataset_name=None):
    train_ds = TextMatchingDataset('train', tokenizer, max_len)
    if test_dataset_name == None:
        eval_job = TextDatasetDataset('eval', tokenizer, max_len)
    else:
        eval_job = TextDatasetDataset(test_dataset_name, tokenizer, max_len)
    eval_task = TextDatasetDataset('task', tokenizer, max_len)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(eval_job, batch_size=batch_size, shuffle=False), DataLoader(eval_task, batch_size=batch_size, shuffle=False)