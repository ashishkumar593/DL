#!/usr/bin/env python3
"""
sensor_simulator.py — improved simulator that sends PPG samples to a server endpoint.

Usage:
    python sensor_simulator.py --url http://localhost:5000/ingest --rate 17 --batch 50 --hr 72 --spo2 98 --device sim-01 --verbose

Features:
 - Produces synthetic IR + Red PPG samples
 - Sends batches as JSON payloads: {"device_id": "...", "samples":[{t,ir,red},...]}
 - Prints server responses and errors
 - Retries transient connection errors
"""

import time, math, random, argparse, requests, sys

def make_ppg_sample(t, hr_bpm=72, spo2=98, noise=1.0, motion=0.3):
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
    return {'t': time.time(), 'ir': float(ir), 'red': float(red)}

def send_payload(url, payload, timeout=3.0, max_retries=2, verbose=False):
    last_exc = None
    for attempt in range(1, max_retries+1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if verbose:
                print(f"[simulator] Server returned {r.status_code}: {r.text}")
            return r.status_code, r.text
        except Exception as e:
            last_exc = e
            if verbose:
                print(f"[simulator] POST attempt {attempt} failed: {e}")
            time.sleep(0.2 * attempt)
    # all retries failed
    if verbose:
        print(f"[simulator] All {max_retries} POST attempts failed; last error: {last_exc}")
    return None, str(last_exc)

def run(url, rate=17.0, batch=50, hr=72.0, spo2=98.0, noise=1.0, motion=0.3, device_id='sim-01', verbose=False):
    interval = 1.0 / float(rate)
    buffer = []
    start = time.time()
    sample_count = 0
    print(f"[simulator] Sending to {url} @ {rate} Hz, batch={batch}  device={device_id}")
    try:
        while True:
            t_rel = time.time() - start
            sample = make_ppg_sample(t_rel, hr_bpm=hr, spo2=spo2, noise=noise, motion=motion)
            buffer.append(sample)
            sample_count += 1

            if len(buffer) >= batch:
                # build payload BEFORE trying network calls
                payload = {'device_id': device_id, 'samples': buffer.copy()}
                status, text = send_payload(url, payload, timeout=3.0, max_retries=3, verbose=verbose)
                # clear buffer after attempt (we don't retry exact same buffer forever)
                buffer = []
            # sleep to simulate sample rate
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[simulator] Stopped by user (KeyboardInterrupt)")
    except Exception as e:
        print(f"[simulator] Fatal error: {e}", file=sys.stderr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Server ingest URL (e.g. http://localhost:5000/ingest)")
    p.add_argument("--rate", type=float, default=17.0, help="Samples per second")
    p.add_argument("--batch", type=int, default=50, help="Samples per POST")
    p.add_argument("--hr", type=float, default=72.0, help="Simulated heart rate (bpm)")
    p.add_argument("--spo2", type=float, default=98.0, help="Simulated SpO2 (%)")
    p.add_argument("--noise", type=float, default=1.0, help="Noise amplitude")
    p.add_argument("--motion", type=float, default=0.3, help="Motion artifact amplitude")
    p.add_argument("--device", dest="device", default="sim-01", help="Device ID string")
    p.add_argument("--verbose", action="store_true", help="Print server responses and retry info")
    args = p.parse_args()
    run(args.url, rate=args.rate, batch=args.batch, hr=args.hr, spo2=args.spo2, noise=args.noise, motion=args.motion, device_id=args.device, verbose=args.verbose)
