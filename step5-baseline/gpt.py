import requests 
import json 
url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions" 
headers = { 
"Content-Type": "application/json", 
"Authorization": "Bearer Key" 
} 
data = { 
"model": "gpt-3.5-turbo", 
"messages": [{"role": "user", "content": "this is a test!"}], 
"temperature": 0.7 
} 
response = requests.post(url, headers=headers, data=json.dumps(data)) 
print(response.json())
