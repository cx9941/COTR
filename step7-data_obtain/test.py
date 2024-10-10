# import torch
#
# # 创建一个示例 tensor
# tensor = torch.tensor([1.0, 3.5, 2.2, 8.7, 4.6, 7.3])
#
# # 使用 topk 函数获取最大值的 n 个元素及其索引
# n = 3
# values, indices = torch.topk(tensor, n)
#
# print("最大值的 n 个元素:", values)
# print("对应的索引:", indices)

# import json
# import pandas as pd
#
# # data1 = pd.read_excel('/home/data/qinchuan/TMIS/paper_code/task_data/eu/eu-bert.xlsx')
# data2 = pd.read_csv('/home/data/qinchuan/TMIS/paper_code/task_data/en/en-gpt.csv')
# tmpindex = list(data2['job_description'])
# data = []
# for index in range(len(data2)):
#     text = data2['job_description'][index]
#     label = data2['task'][index]
#     tmp = {'text': text, 'label':label}
#     for x in range(100):
#         tmpind = tmpindex.index(text)
#         tmplist = eval(data2['bert_task'][tmpind])
#         tmp[f'top{x + 1}'] = tmplist[x]
#     data.append(tmp)
# with open('/home/data/qinchuan/TMIS/paper_code/task_data/en/rank_bert.json', 'w') as f:
#     json.dump(data, f)
# print(len(data))


# allnum = len(data2['task'])
# correct_num_5 = 0
# correct_num_10 = 0
# for i in range(len(data2['task'])):
#     tmplabel = data2['task'][i]
#     tmpdata = eval(data2['gpt_task'][i])
#     if tmplabel in tmpdata[:5]:correct_num_5 += 1
#     if tmplabel in tmpdata[:10]:correct_num_10 += 1
# print(correct_num_5/allnum)
# print(correct_num_10/allnum)


import difflib
def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

print(string_similar("I love Python programming", "Python programming is great"))
