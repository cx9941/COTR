import pandas as pd
import os
import Levenshtein
from utils import word_level_edit_distance
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='en')
parser.add_argument('--process_idx', type=int, default=0)
parser.add_argument('--process_num', type=int, default=6)
args = parser.parse_args()


candidate_jd = pd.read_csv(f'data/{args.dataset_name}/candidate_jd.csv')
target_jd = pd.read_csv(f'data/{args.dataset_name}/target_jd.csv')
candidate_jd['text'] = candidate_jd['job_title'] + ':' + candidate_jd['job_description']
target_jd['text'] = target_jd['job_title'] + ':' + target_jd['job_description']

if not os.path.exists(f'outputs/{args.dataset_name}'):
    os.makedirs(f'outputs/{args.dataset_name}')

import numpy as np
from tqdm import tqdm
sim_matrix = np.zeros([len(target_jd), len(candidate_jd)])
for i in tqdm(range(len(target_jd))):
    for j in range(len(candidate_jd)):
        # sim_matrix[i][j] = Levenshtein.distance(target_jd['text'][i], candidate_jd['text'][j])
        sim_matrix[i][j] = word_level_edit_distance(target_jd['text'][i], candidate_jd['text'][j])
np.save(f'outputs/{args.dataset_name}_sim-word_Levenshtein.npy', sim_matrix)