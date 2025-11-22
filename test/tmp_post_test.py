import json
import http.client

conn = http.client.HTTPConnection("127.0.0.1", 5000, timeout=10)
payload = json.dumps({"train_data": [1,2,3], "ex": 0.001, "ey": 0.001})
headers = {"Content-Type": "application/json"}
try:
    conn.request("POST", "/api/jm/train", payload, headers)
    res = conn.getresponse()
    body = res.read().decode('utf-8')
    print(res.status)
    print(body)
except Exception as e:
    print('ERROR', e)
finally:
    conn.close()
