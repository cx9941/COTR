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
    job_model = BERTForTextMatching(args.bert_model).to(args.device)
    task_model = BERTForTextMatching(args.bert_model).to(args.device)

    # Optimizer
    optimizer = AdamW(list(job_model.parameters()) + list(task_model.parameters()), lr=args.learning_rate)

    # Trainer
    trainer = Trainer(job_model, task_model, optimizer=optimizer, device=args.device)

    # Training loop
    eval_metrics = trainer.evaluate(eval_job, eval_task)
    logging.info(f"\n{pd.DataFrame(eval_metrics).T}")
    best_metrics = {"hits": {20:0}}
    for epoch in range(args.epochs):
        logging.info(f'\nEpoch {epoch + 1}/{args.epochs}')
        train_loss = trainer.train(train_loader)
        logging.info(f'\nTraining loss: {train_loss}')
        eval_metrics = trainer.evaluate(eval_job, eval_task)
        logging.info(f"\n{pd.DataFrame(eval_metrics).T}")
        if eval_metrics['hits'][20] > best_metrics['hits'][20]:
            best_metrics = eval_metrics
            torch.save(trainer.job_model, f'{args.checkpoint_path}/job_model.pt')
            torch.save(trainer.task_model, f'{args.checkpoint_path}/task_model.pt')
            pd.DataFrame(eval_metrics).T.to_csv(f'{args.checkpoint_path}/eval_metrics.csv',sep='\t')


if __name__ == '__main__':
    main()