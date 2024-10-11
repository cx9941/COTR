import numpy as np
from collections import defaultdict
import os
import re

def get_lastest_checkpoint(output_dir):
    latest_epoch = -1
    for i in os.listdir(output_dir):
        if 'checkpoint' in i:
            epoch = int(re.findall(r'checkpoint-(\d+)', i)[0])
            if epoch > latest_epoch:
                latest_epoch = epoch
    if latest_epoch == -1:
        return None
    else:
        return f"{output_dir}/checkpoint-{latest_epoch}"

def gen(x, dataset_name='en'):
    if dataset_name == 'jp':
        ans = f"### クエリ: 100 個の特定のタスクの説明と 1 つの特定のポジションが与えられます。\n"
    else:
        ans = f"### Query: Now given 100 specific task descriptions and 1 specific position, \n"
    name = "".join([f"class{j+1}." + x['bert_task'][j] for j in range(100)])
    text =  x['job_description']

    if dataset_name == 'jp':
        ans += f"具体的なタスク: {name}\n 対応する責任: {text}\n上記の 100 個のタスク記述から、そのポジションの対応する責任に最も一致する 10 個のタスク記述を選択してください (適合度の高い順に並べ替えられています)。\n回答では、次の形式に厳密に従ってください: \n1.xxxx.\n2.xxxx.\n3.xxxx.\n ### 回答:"
    else:
        ans += f'Specific tasks: {name}\n corresponding responsibilities: {text}\nPlease select the 10 task descriptions that best match the corresponding responsibilities of the position from the above 100 task descriptions (sorted from high to low in terms of degree of conformity). \nPlease strictly follow the following format for your answer: \n1.xxxx.\n2.xxxx.\n3.xxxx.\n ### Answer:'
    return ans

def aggregate_expert_opinions(expert_lists, weight_scheme='linear',num_categories=10, top_num=5):
    """
    Aggregates multiple expert opinions to give the most suitable classification result.
    
    Parameters:
    expert_lists (list of list of str): List of lists where each sublist contains the top 10 categories 
                                        as ranked by each expert (from most suitable to least suitable).
    weight_scheme (str): The weighting scheme to use. Can be 'linear' or 'exponential'.
                         'linear' - assigns linearly decreasing weights (10 for 1st, 9 for 2nd, ..., 1 for 10th).
                         'exponential' - assigns exponentially decreasing weights (e.g., 2^9 for 1st, 2^8 for 2nd, ..., 2^0 for 10th).
                         
    Returns:
    str: The most suitable category based on expert opinions.
    """
    
    # Generate weights based on the selected scheme
    if weight_scheme == 'linear':
        weights = np.arange(num_categories, 0, -1)  # e.g., [10, 9, ..., 1]
    elif weight_scheme == 'exponential':
        weights = 2 ** np.arange(num_categories - 1, -1, -1)  # e.g., [512, 256, ..., 1]
    else:
        raise ValueError("Unsupported weight scheme. Choose 'linear' or 'exponential'.")
    
    # Dictionary to accumulate scores for each category
    category_scores = defaultdict(float)
    
    # Aggregate the weights for each category based on expert opinions
    for expert_list in expert_lists:
        for i, category in enumerate(expert_list):
            category_scores[category] += weights[i]
    
    # Find the category with the highest score
    most_suitable_category = max(category_scores, key=category_scores.get)

    most_suitable_scores = category_scores[most_suitable_category]

    top5_keys = sorted(category_scores, key=category_scores.get, reverse=True)[:top_num]
    top5_scores = [category_scores[_] for _ in top5_keys]
    
    return top5_keys, top5_scores

if __name__ == '__main__':
    # Example usage:
    expert_lists = [
        ['cat_A', 'cat_B', 'cat_C', 'cat_D', 'cat_E', 'cat_F', 'cat_G', 'cat_H', 'cat_I', 'cat_J'],
        ['cat_B', 'cat_C', 'cat_A', 'cat_E', 'cat_D', 'cat_F', 'cat_H', 'cat_I', 'cat_G', 'cat_J'],
        ['cat_A', 'cat_D', 'cat_B', 'cat_E', 'cat_C', 'cat_F', 'cat_G', 'cat_H', 'cat_I', 'cat_J']
    ]

    result, scores = aggregate_expert_opinions(expert_lists, weight_scheme='linear')
    print("Most suitable category:", result)