import requests
import json

url = 'http://183.174.229.156:8080'
headers = {
    'Content-Type': 'application/json'
}
data = {
    'prompt': '你好',
    'history': []
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response)
print(response.text)
