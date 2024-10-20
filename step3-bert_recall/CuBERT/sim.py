import torch
import pandas as pd
from transformers import BertTokenizer
from config import get_args
from model import BERTForTextMatching
from dataset import create_data_loader
from trainer import Trainer
import logging
from torch.optim import AdamW
import numpy as np

def main():
    args = get_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    _, eval_job, eval_task = create_data_loader(tokenizer, args.max_len, args.batch_size, args.dataset_name, args.mode)

    job_model = torch.load(f"{args.checkpoint_path}/job_model.pt")
    task_model = torch.load(f"{args.checkpoint_path}/task_model.pt")

    trainer = Trainer(job_model, task_model, args.device)
    similarity_matrix = trainer.compute_sim(eval_job, eval_task)
    np.save(f"{args.checkpoint_path}/similarity_matrix.npy", similarity_matrix)

if __name__ == '__main__':
    main()