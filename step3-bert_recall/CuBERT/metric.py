import numpy as np

# 计算余弦相似度
def cosine_similarity_matrix(jobs, tasks):
    job_norms = np.linalg.norm(jobs, axis=1, keepdims=True)
    task_norms = np.linalg.norm(tasks, axis=1, keepdims=True)
    
    similarity_matrix = np.dot(jobs, tasks.T) / (job_norms * task_norms.T)
    return similarity_matrix

# 获取排名
def get_ranking(similarity_matrix):
    rankings = np.argsort(-similarity_matrix, axis=1)  # 从大到小排序
    return rankings

def compute_metrics(rankings, labels, k_values=[1, 3, 5, 10, 20]):
    num_jobs = len(labels)
    recalls = {k: 0 for k in k_values}
    precisions = {k: 0 for k in k_values}
    hits = {k: 0 for k in k_values}
    
    for i in range(num_jobs):
        true_label = labels[i]
        predicted_labels = rankings[i]
        
        for k in k_values:
            top_k_predictions = predicted_labels[:k]
            if true_label in top_k_predictions:
                hits[k] += 1
                recalls[k] += 1
                precisions[k] += 1.0 / k
    
    recall_scores = {k: recalls[k] * 100 / num_jobs for k in k_values}
    precision_scores = {k: precisions[k]  * 100  / num_jobs for k in k_values}
    hit_scores = {k: hits[k] * 100  / num_jobs for k in k_values}
    
    return recall_scores, precision_scores, hit_scores

# 主函数
def calculate_recall_precision_hits(all_job_embeddings, all_task_embeddings, all_labels):
    # Step 1: 计算余弦相似度
    similarity_matrix = cosine_similarity_matrix(all_job_embeddings, all_task_embeddings)
    
    # Step 2: 获取排名
    rankings = get_ranking(similarity_matrix)
    
    # Step 3: 计算 recall, precision 和 hits
    recall_scores, precision_scores, hit_scores = compute_metrics(rankings, all_labels, k_values=[1, 3, 5, 10, 20, 50, 100, 200])
    
    return recall_scores, precision_scores, hit_scores

if __name__ == '__main__':
    recall_scores, precision_scores, hit_scores = calculate_recall_precision_hits(all_job_embeddings, all_task_embeddings, all_labels)

    print(f"Recall@1: {recall_scores[1]}, Recall@3: {recall_scores[3]}, Recall@5: {recall_scores[5]}")
    print(f"Precision@1: {precision_scores[1]}, Precision@3: {precision_scores[3]}, Precision@5: {precision_scores[5]}")
    print(f"Hits@1: {hit_scores[1]}, Hits@3: {hit_scores[3]}, Hits@5: {hit_scores[5]}")