import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "CMRR": 45,
    "PRSUP": 30,
    "depth_of_ cover": 220,
    "intersection_diagonal": 5.2,
    "mining_hight": 2.8
}

res = requests.post(url, json=data)
print(res.json())
