import numpy as np
from collections import defaultdict

def aggregate_expert_opinions(expert_lists, weight_scheme='linear',num_categories=10):
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
    
    return most_suitable_category, most_suitable_scores

if __name__ == '__main__':
    # Example usage:
    expert_lists = [
        ['cat_A', 'cat_B', 'cat_C', 'cat_D', 'cat_E', 'cat_F', 'cat_G', 'cat_H', 'cat_I', 'cat_J'],
        ['cat_B', 'cat_C', 'cat_A', 'cat_E', 'cat_D', 'cat_F', 'cat_H', 'cat_I', 'cat_G', 'cat_J'],
        ['cat_A', 'cat_D', 'cat_B', 'cat_E', 'cat_C', 'cat_F', 'cat_G', 'cat_H', 'cat_I', 'cat_J']
    ]

    result = aggregate_expert_opinions(expert_lists, weight_scheme='linear')
    print("Most suitable category:", result)