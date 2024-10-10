# 1.Available Models: gpt-4, gpt-4-32k, gpt-3.5-turbo, gpt-3.5-turbo-16k
# 2.Request Instruction:

import requests 
import json 
import time

def get_gpt_response(text):
    url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions" 
    headers = { 
    "Content-Type": "application/json", 
    "Authorization": "6bb1129763df406b8e856d1a072db033a541b784dd454eceaea00492a34bb6a5" 
    } 
    data = { 
    "model": "gpt-3.5-turbo", 
    "messages": [{"role": "user", "content": text}], 
    "temperature": 0.7 
    } 
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data)) 
        ans = response.json()
        ans = ans['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        time.sleep(60)
        ans = get_gpt_response(text)
    time.sleep(30)
    return ans

# 调用 EB4.0 
def get_eb_response(prompt):
    data = {
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.6,
        "top_p": 0.8
    }

    headers ={
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    # eb3.5
    # url = "ERNIE-Speed-128K"
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=24.e0de31bbbb17f0408ee975b93727be5b.2592000.1729749764.282335-115680234"

    # eb4
    # url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=24.e0de31bbbb17f0408ee975b93727be5b.2592000.1729749764.282335-115680234"
    response = requests.request("POST", url, headers=headers, data=json.dumps(data))
    return response.json()['result']


import os
import qianfan
os.environ["QIANFAN_ACCESS_KEY"] = "ALTAKAPLmViDOdQsF3NMPPQtx6"
os.environ["QIANFAN_SECRET_KEY"] = "61a662aab7314cecb20a122a77109fcd"

def get_ebot_results(prompt):
    try:
        chat_comp = qianfan.ChatCompletion()

        resp = chat_comp.do(model="ERNIE-Speed-128K", messages=[{
            "role": "user",
            "content": prompt
        }])
        ans = resp["body"]['result']
    except Exception as e:
        print(e)
        ans = get_ebot_results(prompt)
        time.sleep(60)
    time.sleep(10)
    return ans

if __name__ == '__main__':
    ans = get_ebot_results('你好')
    print(ans)