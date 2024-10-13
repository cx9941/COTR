import re
import pandas as pd
import json
import Levenshtein
import argparse
import difflib
import os
from tqdm import tqdm
from config import args

if os.path.exists(args.metric_path):
    exit()

def closest_string_by_levenshtein(A, string_list):
    # 初始化最小距离和最近的字符串
    closest_string = None
    min_distance = float('inf')

    # 遍历列表中的每个字符串
    for s in string_list:
        # 计算 Levenshtein 编辑距离
        edit_distance = Levenshtein.distance(A, s)

        # 更新最近的字符串和最小距离
        if edit_distance < min_distance:
            min_distance = edit_distance
            closest_string = s

    if min_distance > 5:closest_string = ''
    return closest_string

def string_similar(A, string_list):
    closest_string = None
    min_distance = 0

    for s in string_list:
        # 计算 Levenshtein 编辑距离
        edit_distance = difflib.SequenceMatcher(None, A, s).quick_ratio()

        # 更新最近的字符串和最小距离
        if edit_distance > min_distance:
            min_distance = edit_distance
            closest_string = s

    return closest_string
result = pd.read_csv(args.data_path)
# result = result[(result['tag']==1) | (result['score']>args.thred)]
result = result[(result['tag']==1)].reset_index(drop=True)

data = [open(f"{args.output_dir}/{i}.txt", 'r').read()  for i in range(len(os.listdir(args.output_dir)))][:len(result)]


labeldata = pd.read_csv(args.label_path, sep='\t')
exist_label = list(labeldata['DWA Title'])
exist_dict = {}
for i in range(len(exist_label)):
    exist_dict[exist_label[i]] = i


index_list = {}
correct_5 = 0
correct_10 = 0
for index, i in tqdm(enumerate(data), total=len(data)):
    # text = re.search(pattern, i)

    # 使用 findall 提取所有匹配的内容
    tag = True
    matches = i[i.find('Answer:'):].split('\n')
    tmpexist = []
    allexist = []
    tmpnum = 0
    for match in matches:
        match = match.split('.')
        if len(match) == 1:continue
        # print(match)
        match = match[-2]
        if args.mapping == 'le':
            match = closest_string_by_levenshtein(match, exist_label)
        else:
            match = string_similar(match, exist_label)
        if match == '':continue
        if exist_dict.get(match):
            if exist_dict[match] in allexist: continue
            tmpnum += 1
            allexist.append(exist_dict[match])
            if match != result['task'][index]:
                tmpexist.append(exist_dict[match])
            else:
                if tag:
                    correct_5 += 1
                correct_10 += 1
        if tmpnum >=5:tag=False
        if tmpnum >=10:break
    index_list[index] = tmpexist

hits_5 = correct_5 / len(data)
hits_10 = correct_10 / len(data)
ans = {'hits@5': hits_5, 'hits@10': hits_10}
print(ans)
json.dump(ans, open(args.metric_path, 'w'))