import re
import pandas as pd
import json
import Levenshtein
import argparse
import difflib

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/data/qinchuan/TMIS/paper_code/task_data/eu/eu-gpt.csv', type=str,
                        help='')
    parser.add_argument('--result_path', default='/home/data/qinchuan/TMIS/paper_code/output/data_obtain/baichuan/infer_sft/0.csv',
                        type=str,
                        help='')
    parser.add_argument('--label_path',
                        default='/home/data/qinchuan/TMIS/COTR/step3-bert_recall/data/eu/task.csv',
                        type=str, help='')
    parser.add_argument('--mapping',
                        default='le', choices=['le', 'df'],
                        type=str, help='')
    return parser.parse_args()


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


args = set_args()
data = pd.read_csv(args.result_path)
result = pd.read_csv(args.data_path)

labeldata = pd.read_csv(args.label_path, sep='\t')
exist_label = list(labeldata['DWA Title'])
exist_dict = {}
for i in range(len(exist_label)):
    exist_dict[exist_label[i]] = i


index_list = {}
correct = 0
for index, i in enumerate(data['response_sft']):
    # text = re.search(pattern, i)

    # 使用 findall 提取所有匹配的内容
    matches = i.split('Answer:')[1].strip().split('\n')
    tmpexist = []
    allexist = []
    tmpnum = 0
    for match in matches:
        match = match.split('.')
        if len(match) == 1:continue
        print(match)
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
                correct += 1
        if tmpnum >=5:break
    index_list[index] = tmpexist

# with open('/home/data/qinchuan/TMIS/paper_code/banking_data/gpt_index.json', 'w') as f:
#     json.dump(index_list, f)
print(correct / len(data))
