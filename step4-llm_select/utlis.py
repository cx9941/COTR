import numpy as np
from collections import defaultdict

def gen(x, dataset_name='en'):
    if dataset_name == 'jp':
        ans = f"100 個の具体的なタスクの説明と 1 つの具体的なポジションが与えられます。\n"
    else:
        ans = f"Now given 100 specific task descriptions and 1 specific position, \n"
    for i in range(100):
        ans += f"{i}." + x[f"task{i}"] + "\n"

    if dataset_name == 'jp':
        ans += f"対応する責任:\n{x['job_description']}\n"
        ans += f"上記の 100 個のタスクの説明から、職務責任に最も適合する 10 個のタスクの説明を選択します (タスクと責任の間の相関関係によって並べられ、相関関係が高いほど上位にランクされます)。\n回答には、次の形式を厳守してください: \n1.xxxx. (タスク x)\n2.xxxx. (タスク x)\n3.xxxx. (タスク x)\n"
    else:
        ans += f"corresponding responsibilities:\n{x['job_description']}\n"
        ans += "Please select the 10 task descriptions that best match the corresponding responsibilities of the position from the above 100 task descriptions (sorted from high to low in terms of degree of conformity). \nPlease strictly follow the following format for your answer: \n1.xxxx. (Task x)\n2.xxxx. (Task x)\n3.xxxx. (Task x)\n"
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