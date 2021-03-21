import request

url = 'http://localhost:5000/results'
r = requests.post(url, json={'image':5)
print(r.json())

