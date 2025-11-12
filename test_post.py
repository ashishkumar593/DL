# test_post.py
import requests, time

url = "http://localhost:5000/ingest"
payload = {
    "device_id": "ping",
    "samples": [{"t": time.time(), "ir": 1.02, "red": 0.98} for _ in range(10)],
}

print(f"Posting to {url} ...")
try:
    r = requests.post(url, json=payload, timeout=5)
    print("Response code:", r.status_code)
    print("Response text:", r.text)
except Exception as e:
    print("Request failed:", e)
