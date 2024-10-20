import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
from metric import calculate_recall_precision_hits, cosine_similarity_matrix, get_ranking
from utils import whitening_torch_final

class Trainer:
    def __init__(self, job_model, task_model, device, optimizer=None, bert_mode='normal',k=16):
        self.job_model = job_model
        self.task_model = task_model
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CosineEmbeddingLoss()
        self.bert_mode = bert_mode
        self.k = k

    def train(self, data_loader):
        self.job_model.train()
        self.task_model.train()
        total_loss = 0

        with tqdm(data_loader) as bar:

            for batch in bar:
                job_input_ids = batch['job_input_ids'].to(self.device)
                job_attention_mask = batch['job_attention_mask'].to(self.device)
                task_input_ids = batch['task_input_ids'].to(self.device)
                task_attention_mask = batch['task_attention_mask'].to(self.device)

                job_embeddings = self.job_model(job_input_ids, job_attention_mask)
                task_embeddings = self.task_model(task_input_ids, task_attention_mask)


                # if self.bert_mode == 'white':
                #     job_embeddings, task_embeddings = whitening_torch_final(job_embeddings, task_embeddings, k=self.k)

                batch_size = job_embeddings.size(0)

                # 正样本 (label=1)，拉近它们的表征
                positive_labels = torch.ones(batch_size).to(self.device)
                positive_loss = self.criterion(job_embeddings, task_embeddings, positive_labels)

                # 构造负样本（不同的 job 和 task 进行配对）
                neg_indices = torch.randperm(batch_size)
                negative_task_embeddings = task_embeddings[neg_indices]

                # 负样本 (label=-1)，拉远它们的表征
                negative_labels = -torch.ones(batch_size).to(self.device)
                negative_loss = self.criterion(job_embeddings, negative_task_embeddings, negative_labels)

                # 总损失是正样本和负样本的损失之和
                loss = positive_loss + negative_loss
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                bar.set_description(f"loss: {loss.item()}")

        return total_loss / len(data_loader)

    def evaluate(self, eval_job, eval_task):
        self.job_model.eval()
        self.task_model.eval()

        all_job_embeddings = []
        all_task_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch, label in tqdm(eval_job):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                job_embeddings = self.job_model(input_ids, attention_mask)
                all_job_embeddings.append(job_embeddings)
                all_labels.append(label)

            for batch in tqdm(eval_task):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                task_embeddings = self.task_model(input_ids, attention_mask)
                all_task_embeddings.append(task_embeddings)
        
        all_job_embeddings = torch.concat(all_job_embeddings)
        all_task_embeddings = torch.concat(all_task_embeddings)

        if self.bert_mode == 'white':
            all_job_embeddings, all_task_embeddings = whitening_torch_final(all_job_embeddings, all_task_embeddings, k=self.k)

        all_job_embeddings = all_job_embeddings.detach().cpu().numpy()
        all_task_embeddings = all_task_embeddings.detach().cpu().numpy()

        # all_job_embeddings = torch.concat(all_job_embeddings).detach().cpu().numpy()
        # all_task_embeddings = torch.concat(all_task_embeddings).detach().cpu().numpy()
        all_labels = torch.concat(all_labels).detach().cpu().numpy()

        recall_scores, precision_scores, hit_scores = calculate_recall_precision_hits(all_job_embeddings, all_task_embeddings, all_labels)
        final_metrics = {'recall': recall_scores, "precision": precision_scores, "hits": hit_scores}

        return final_metrics
    

    def compute_sim(self, eval_job, eval_task):
        self.job_model.eval()
        self.task_model.eval()

        all_job_embeddings = []
        all_task_embeddings = []

        with torch.no_grad():
            for batch in tqdm(eval_job):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                job_embeddings = self.job_model(input_ids, attention_mask)
                all_job_embeddings.append(job_embeddings)

            for batch in tqdm(eval_task):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                task_embeddings = self.task_model(input_ids, attention_mask)
                all_task_embeddings.append(task_embeddings)

        all_job_embeddings = torch.concat(all_job_embeddings)
        all_task_embeddings = torch.concat(all_task_embeddings)

        if self.bert_mode == 'white':
            all_job_embeddings, all_task_embeddings = whitening_torch_final(all_job_embeddings, all_task_embeddings, k=self.k)

        all_job_embeddings = all_job_embeddings.detach().cpu().numpy()
        all_task_embeddings = all_task_embeddings.detach().cpu().numpy()
        
        similarity_matrix = cosine_similarity_matrix(all_job_embeddings, all_task_embeddings)

        return similarity_matrix

    def test(self, eval_job, eval_task):
        self.job_model.eval()
        self.task_model.eval()

        all_job_embeddings = []
        all_task_embeddings = []

        with torch.no_grad():
            for batch in tqdm(eval_job):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                job_embeddings = self.job_model(input_ids, attention_mask)
                all_job_embeddings.append(job_embeddings)

            for batch in tqdm(eval_task):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                task_embeddings = self.task_model(input_ids, attention_mask)
                all_task_embeddings.append(task_embeddings)

        all_job_embeddings = torch.concat(all_job_embeddings)
        all_task_embeddings = torch.concat(all_task_embeddings)

        if self.bert_mode == 'white':
            all_job_embeddings, all_task_embeddings = whitening_torch_final(all_job_embeddings, all_task_embeddings, k=self.k)
        
        all_job_embeddings, all_task_embeddings = all_job_embeddings.detach().cpu().numpy(), all_task_embeddings.detach().cpu().numpy()
        
        similarity_matrix = cosine_similarity_matrix(all_job_embeddings, all_task_embeddings)
        rankings, sort_similarity_matrix = get_ranking(similarity_matrix)

        return sort_similarity_matrix, rankings