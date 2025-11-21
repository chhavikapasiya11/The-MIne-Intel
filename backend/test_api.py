import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "CMRR": 80,
    "PRSUP": 10,
    "depth_of_ cover": 100,
    "intersection_diagonal": 1.2,
    "mining_hight": 5
}

res = requests.post(url, json=data)
print(res.json())
