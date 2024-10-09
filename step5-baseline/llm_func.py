# 1.Available Models: gpt-4, gpt-4-32k, gpt-3.5-turbo, gpt-3.5-turbo-16k
# 2.Request Instruction:

import requests 
import json 

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
    response = requests.post(url, headers=headers, data=json.dumps(data)) 
    ans = response.json()
    return ans['choices'][0]['message']['content']

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

def get_ebot_results(prompt):
    import qianfan
    # 通过环境变量初始化认证信息
    # 方式一：【推荐】使用安全认证AK/SK鉴权
    # 替换下列示例中参数，安全认证Access Key替换your_iam_ak，Secret Key替换your_iam_sk，如何获取请查看https://cloud.baidu.com/doc/Reference/s/9jwvz2egb
    os.environ["QIANFAN_ACCESS_KEY"] = "ALTAKAPLmViDOdQsF3NMPPQtx6"
    os.environ["QIANFAN_SECRET_KEY"] = "61a662aab7314cecb20a122a77109fcd"
    chat_comp = qianfan.ChatCompletion()
    # 指定特定模型
    resp = chat_comp.do(model="ERNIE-Speed-8K", messages=[{
        "role": "user",
        "content": "你好"
    }])
    return resp["body"] 

if __name__ == '__main__':
    ans = get_gpt_response('你好')
    print(ans)