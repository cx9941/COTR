import re
import pandas as pd
import json

data = pd.read_csv('/home/data/qinchuan/TMIS/paper_code/output/data_obtain/llama/out147/0.csv')
pattern = r'specific text:(.*?)please'
pattern2 = r'\d+\.(.*?)\('

with open('/home/data/qinchuan/TMIS/paper_code/banking_data/categories.json', 'r') as f:
    exist_label = json.load(f)
exist_dict = {}
for i in range(len(exist_label)):
    exist_dict[exist_label[i]] = i


index_list = {}
correct = 0
for index, i in enumerate(data['response_sft']):
    # text = re.search(pattern, i)

    # 使用 findall 提取所有匹配的内容
    matches = re.findall(pattern2, i)
    tmpexist = []
    allexist = []
    tmpnum = 0
    for match in matches[3:]:
        if exist_dict.get(match):
            if exist_dict[match] in allexist: continue
            tmpnum += 1
            allexist.append(exist_dict[match])
            if match != data['label'][index]:
                tmpexist.append(exist_dict[match])
            else:
                correct += 1
        if tmpnum >=5:break
    index_list[index] = tmpexist

with open('/home/data/qinchuan/TMIS/paper_code/banking_data/gpt_index.json', 'w') as f:
    json.dump(index_list, f)
print(correct / len(data))
