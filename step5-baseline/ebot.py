import requests
import json

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

if __name__=="__main__":
    prompt = "介绍下北京"
    response = get_eb_response(prompt)
    print(response)

# https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu