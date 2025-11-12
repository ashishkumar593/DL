# sensor_simulator.py
"""
Simulated IoT sensor that POSTS PPG samples (IR + Red) to a server endpoint.

Usage:
    python sensor_simulator.py --url http://localhost:8501/ingest --rate 17 --batch 50

Parameters:
    --url   : server ingestion endpoint (Streamlit app will run a Flask receiver at /ingest)
    --rate  : samples per second (default 17)
    --batch : how many samples per POST (default 50)
"""

import time, math, random, json, argparse, requests

def make_ppg_sample(t, hr_bpm=72, spo2=98, noise=1.0, motion=0.3):
    # simple synthetic PPG model: baseline DC + AC sinusoid + motion + noise
    dc_ir = 1.0
    dc_red = 0.95
    f = hr_bpm / 60.0
    ac_ir = 0.03 * math.sin(2 * math.pi * f * t) + 0.005 * math.sin(2 * math.pi * 2 * f * t)
    ac_red = 0.028 * math.sin(2 * math.pi * f * t + 0.2) + 0.004 * math.sin(2 * math.pi * 2 * f * t + 0.25)
    motion_component = motion * (0.02 * math.sin(2 * math.pi * 0.5 * t) + 0.015 * math.sin(2 * math.pi * 0.25 * t))
    noise_ir = noise * (random.random() - 0.5) * 0.01
    noise_red = noise * (random.random() - 0.5) * 0.01
    ir = dc_ir + ac_ir + motion_component + noise_ir
    red = dc_red + ac_red + motion_component * 0.9 + noise_red
    return {'t': time.time(), 'ir': ir, 'red': red}

def run(url, rate=17, batch=50, hr=72, spo2=98, noise=1.0, motion=0.3):
    interval = 1.0 / rate
    buffer = []
    start = time.time()
    sample_count = 0
    print(f"[simulator] Sending to {url} @ {rate} Hz, batch={batch}")
    try:
        while True:
            t_rel = time.time() - start
            sample = make_ppg_sample(t_rel, hr_bpm=hr, spo2=spo2, noise=noise, motion=motion)
            buffer.append(sample)
            sample_count += 1
            if len(buffer) >= batch:
                payload = {'device_id': 'sim-01', 'samples': buffer}
                try:
                    r = requests.post(url, json=payload, timeout=2.0)
                    if r.status_code != 200:
                        print(f"[simulator] Server returned {r.status_code}: {r.text}")
                except Exception as e:
                    print(f"[simulator] POST error: {e}")
                buffer = []
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[simulator] Stopped by user")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Server ingest URL (e.g. http://localhost:8501/ingest)")
    parser.add_argument("--rate", type=float, default=17.0, help="Samples per second")
    parser.add_argument("--batch", type=int, default=50, help="Samples per POST")
    parser.add_argument("--hr", type=float, default=72.0, help="Simulated heart rate (bpm)")
    parser.add_argument("--spo2", type=float, default=98.0, help="Simulated SpO2 (%) - used in drift only")
    parser.add_argument("--noise", type=float, default=1.0, help="Noise amplitude")
    parser.add_argument("--motion", type=float, default=0.3, help="Motion artifact amplitude")
    args = parser.parse_args()
    run(args.url, rate=args.rate, batch=args.batch, hr=args.hr, spo2=args.spo2, noise=args.noise, motion=args.motion)
