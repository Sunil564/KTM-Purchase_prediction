import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Screen':2, 'Capacity':3, 'Connectivity':0, 'Gen':0})

print(r.json())