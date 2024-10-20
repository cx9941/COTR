import torch
import pandas as pd
from transformers import AutoTokenizer
from config import get_args
from model import BERTForTextMatching
from dataset import create_data_loader
from trainer import Trainer
import logging
from torch.optim import AdamW

def main():
    args = get_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Data loader
    train_loader, eval_job, eval_task = create_data_loader(tokenizer, args.max_len, args.batch_size, args.dataset_name, args.mode)

    # Models
    job_model = torch.load(f"{args.checkpoint_path}/job_model.pt")
    task_model = torch.load(f"{args.checkpoint_path}/task_model.pt")

    # Optimizer
    optimizer = AdamW(list(job_model.parameters()) + list(task_model.parameters()), lr=args.learning_rate)

    # Trainer
    trainer = Trainer(job_model, task_model, optimizer=optimizer, device=args.device, bert_mode=args.bert_mode, k=args.k)

    # Training loop
    eval_metrics = trainer.evaluate(eval_job, eval_task)
    print(f"\n{pd.DataFrame(eval_metrics).T}")


if __name__ == '__main__':
    main()