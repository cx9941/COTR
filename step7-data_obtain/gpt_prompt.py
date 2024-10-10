import pandas as pd

i= beginindex
r = pd.read_csv("bert_simi.csv")
query = []
while i < endindex:
    print(i)
    task_id= f'vgfhcxzxgrxcghvkjhbbgjv{i}'# 批次号,自己生成尽量保证唯一,推荐用md5/uuid之类的长hash
    text = r.iloc[i]['text']
    name="".join([f"class{j}."+r.iloc[i][f"top{j}_name"]forjin range(1,78)])
    # name="".join([f"class{j}."+label_list[j]forjin range(77)])
    q=f'Now gives 77 descriptions of specific intents and 1 specific text,\n specific class:{name}\n,specific text:{text}\n please sort the above 77 types of intents according to the degree of matching with the intent of the text.\n Please ' \
      f'strictly follow the format of the answer given:\n 1.xxxx(confidence score:xxxxx,reason:xxxxx)\n 2.xxxx(confidence score:xxxxx, reason:xxxxx)\n 3.xxxx(confidence score:xxxxx, reason:xxxxx)\n.'

    # q=f'Now gives 77 descriptions of specific intents and 1 specific text,\n specific class:{name}\n,specific text: {text} \n please sort the above 77 types of intents according to the degree of matching with the intent of the text.\n Please
    # strictly follow the format of the answer given:\n1.xxxx(class x)\n 2.xxxx(class x)\n 3.xxxx(class x)\n.'
    query.append(q)
    with open(f"/individual/fangchuyu/icde/chatgpt_query/querybank/{i}.txt","w")as w:
        w.write(q)
    post_data = {
        "messages":[
            {"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content": q}
        ],
        "model":"gpt-3.5-turbo",
    }
    url='http://spider,weizhipin.com/chatgpt/query?key='+key+'6task_id='+task_id
    try:
        resp=requests.post(url,json=post_data,timeout=3a0)
        data = resp.json()
        if data['error'] ==0:
            num = 0
            content = data['gpt_data']['choices'][0]['message']['content']
            with open(f'/individuaL/fangchuyu/icde/chatgpt_query/returnbank/{i}.txt',"w")as w:
                w.write(content)
            i+=1
        else:
            time.sleep(1)
            print(data)
            num+= 1
            if num>5:
                i+=1
    except:
        continue
