import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "CMRR": 100,
    "PRSUP": 100,
    "depth_of_ cover": 1,
    "intersection_diagonal": 1,
    "mining_hight": 1
}

res = requests.post(url, json=data)
print(res.json())
