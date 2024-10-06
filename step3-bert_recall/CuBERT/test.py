import torch
import pandas as pd
from transformers import BertTokenizer
from config import get_args
from model import BERTForTextMatching
from dataset import create_data_loader
from trainer import Trainer
import logging
from torch.optim import AdamW

def main():
    args = get_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    _, eval_job, eval_task = create_data_loader(tokenizer, args.max_len, args.batch_size, args.test_dataset_name)

    job_model = torch.load(f"{args.checkpoint_path}/job_model.pt")
    task_model = torch.load(f"{args.checkpoint_path}/task_model.pt")

    trainer = Trainer(job_model, task_model, args.device)
    rankings = trainer.test(eval_job, eval_task)

    eval_job_dataset = eval_job.dataset.data
    eval_task_dataset = eval_task.dataset.data

    for i in range(args.top_num):
        candidate_task = eval_task_dataset[['DWA Title']].loc[rankings[:,i]].reset_index(drop=True)
        candidate_task.columns = [f'task{i}']
        eval_job_dataset = pd.concat([eval_job_dataset, candidate_task], axis=1)
    eval_job_dataset.to_csv(f"{args.result_path}/{args.test_dataset_name}.csv", index=None, sep='\t')
    eval_job_dataset.to_excel(f"{args.result_path}/{args.test_dataset_name}.xlsx", index=None)

if __name__ == '__main__':
    main()