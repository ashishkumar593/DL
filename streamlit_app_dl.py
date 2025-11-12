# streamlit_app_dl.py
"""
Streamlit dashboard with DL inference.
Requires model.pt produced by train_dl.py

Run:
    pip install streamlit flask numpy pandas plotly torch requests
    # Train model first:
    python train_dl.py --epochs 20 --batch 64 --save-path model.pt
    # Then start Streamlit:
    streamlit run streamlit_app_dl.py
"""
import streamlit as st
from threading import Thread
from queue import Queue, Empty
from flask import Flask, request, jsonify
import time, numpy as np, pandas as pd
import plotly.graph_objs as go
import os

# DL imports
import torch
from model import PPG1DCNN

# --------- Config ----------
MODEL_PATH = "model.pt"
FS_DEFAULT = 17
WINDOW_S_DEFAULT = 8.0

# --------- Shared queue ----------
incoming_queue = Queue(maxsize=20000)

# --------- Flask receiver ----------
def create_flask_app(queue: Queue):
    app = Flask("receiver")
    @app.route("/ingest", methods=["POST"])
    def ingest():
        data = request.get_json()
        if not data or 'samples' not in data:
            return jsonify({"error": "invalid payload"}), 400
        samples = data['samples']
        device_id = data.get('device_id', 'unknown')
        try:
            for s in samples:
                incoming = {'device_id': device_id, 't': float(s.get('t', time.time())), 'ir': float(s['ir']), 'red': float(s['red'])}
                queue.put(incoming, block=False)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"status": "ok", "received": len(samples)}), 200
    return app

def run_flask_in_thread(queue):
    app = create_flask_app(queue)
    def run():
        app.run(host="0.0.0.0", port=8501, debug=False, use_reloader=False)
    t = Thread(target=run, daemon=True)
    t.start()
    return t

# --------- Simple classical processing (same as earlier) ----------
def detect_peaks_simple(sig, min_distance_s=0.3, fs=17, threshold=None):
    sig = np.asarray(sig)
    if len(sig) < 3:
        return []
    if threshold is None:
        threshold = sig.mean() + 0.25*sig.std()
    peaks = []
    for i in range(1, len(sig)-1):
        if sig[i] > sig[i-1] and sig[i] > sig[i+1] and sig[i] > threshold:
            peaks.append(i)
    min_samples = int(min_distance_s * fs)
    cleaned = []
    for p in peaks:
        if not cleaned or (p - cleaned[-1]) >= min_samples:
            cleaned.append(p)
    return cleaned

def estimate_spo2_window(ir_vals, red_vals):
    if len(ir_vals) < 5 or len(red_vals) < 5:
        return None
    ac_ir = np.max(ir_vals) - np.min(ir_vals)
    ac_red = np.max(red_vals) - np.min(red_vals)
    dc_ir = np.mean(ir_vals)
    dc_red = np.mean(red_vals)
    if dc_ir == 0 or dc_red == 0 or ac_ir == 0:
        return None
    r = (ac_red/dc_red) / (ac_ir/dc_ir)
    spo2 = int(np.clip(int(round(110 - 25*r)), 70, 100))
    return spo2

# --------- Load model (if exists) ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = PPG1DCNN(in_channels=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device).eval()
        print(f"[dl] Loaded model from {MODEL_PATH} on device {device}")
    except Exception as e:
        print("[dl] Failed loading model:", e)
        model = None
else:
    print("[dl] No model found at", MODEL_PATH)

# --------- Streamlit UI ----------
st.set_page_config(page_title="IoT PPG Dashboard (DL)", layout="wide")
st.title("IoT PPG Dashboard (DL) — Receiver + Live Processing")
st.markdown("This app runs a small Flask receiver and performs classical + DL inference on incoming PPG windows. Train a model with `train_dl.py` to enable DL predictions.")

if 'flask_started' not in st.session_state:
    st.session_state.flask_started = False

if not st.session_state.flask_started:
    run_flask_in_thread(incoming_queue)
    st.session_state.flask_started = True
    st.info("Flask ingest endpoint running at POST http://localhost:8501/ingest")

# Controls
with st.sidebar:
    st.header("Controls")
    fs = st.slider("Assumed sample rate (Hz)", min_value=8, max_value=100, value=FS_DEFAULT)
    window_sec = st.slider("Window size (s) for inference", min_value=2, max_value=12, value=int(WINDOW_S_DEFAULT))
    min_peak_distance = st.slider("Min peak distance (ms)", min_value=200, max_value=1000, value=300)
    auto_scroll = st.checkbox("Auto-scroll latest samples", value=True)
    if st.button("Reload DL model"):
        if os.path.exists(MODEL_PATH):
            try:
                loaded = PPG1DCNN(in_channels=2)
                loaded.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                loaded.to(device).eval()
                model = loaded
                st.success("Model reloaded")
            except Exception as e:
                st.error("Failed to reload model: " + str(e))

# In-memory dataframe store
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['device_id','t','ir','red'])

# Drain incoming queue
pulled = 0
while True:
    try:
        item = incoming_queue.get(block=False)
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([item])], ignore_index=True)
        pulled += 1
        if pulled > 5000: break
    except Empty:
        break

# Trim to last N seconds
max_seconds = 300
if not st.session_state.df.empty:
    now = time.time()
    st.session_state.df = st.session_state.df[st.session_state.df['t'] >= (now - max_seconds)].reset_index(drop=True)

cols = st.columns([3,1])
with cols[0]:
    st.subheader("Live PPG (last {:.1f}s)".format(window_sec))
    df_plot = st.session_state.df.copy()
    if df_plot.empty:
        st.write("No data yet. Run the sensor simulator.")
    else:
        latest_t = df_plot['t'].max()
        window_df = df_plot[df_plot['t'] >= (latest_t - window_sec)]
        if window_df.empty:
            st.write("Waiting for more samples to fill the window...")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=window_df['t'], y=window_df['ir'], mode='lines', name='IR'))
            fig.add_trace(go.Scatter(x=window_df['t'], y=window_df['red'], mode='lines', name='Red'))
            fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=360, xaxis_title="time")
            st.plotly_chart(fig, use_container_width=True)

with cols[1]:
    st.subheader("Metrics (classical + DL)")
    if st.session_state.df.empty:
        st.metric("HR (classical)", "-- bpm")
        st.metric("SpO2 (classical)", "-- %")
        st.metric("HR (DL)", "-- bpm")
        st.metric("SpO2 (DL)", "-- %")
    else:
        last_device = st.session_state.df['device_id'].iloc[-1]
        device_df = st.session_state.df[st.session_state.df['device_id']==last_device]
        latest_t = device_df['t'].max()
        window_df = device_df[device_df['t'] >= (latest_t - window_sec)]
        ir_vals = window_df['ir'].astype(float).to_numpy()
        red_vals = window_df['red'].astype(float).to_numpy()

        # Classical estimates
        peaks = detect_peaks_simple(ir_vals, min_distance_s=min_peak_distance/1000.0, fs=fs)
        hr_bpm = None
        if len(peaks) >= 2:
            peak_times = window_df['t'].values[peaks]
            intervals = np.diff(peak_times)
            if len(intervals)>0 and np.mean(intervals)>0:
                hr_bpm = float(np.round(60.0 / np.mean(intervals),1))
        spo2_classical = estimate_spo2_window(ir_vals, red_vals)

        # DL inference (if model available)
        hr_dl = None
        spo2_dl = None
        if model is not None and len(ir_vals) >= int(window_sec*fs):
            # create fixed-length window of L = int(window_sec*fs)
            L = int(window_sec*fs)
            # take most recent L samples (if more, slice)
            ir_win = ir_vals[-L:]
            red_win = red_vals[-L:]
            # normalize similar to training: per-channel zero-mean unit-std
            irn = (ir_win - np.mean(ir_win)) / (np.std(ir_win) + 1e-8)
            redn = (red_win - np.mean(red_win)) / (np.std(red_win) + 1e-8)
            x = np.stack([irn, redn], axis=0)[None, ...]  # shape (1,2,L)
            xt = torch.tensor(x, dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model(xt).cpu().numpy().squeeze()
                hr_dl = float(out[0])
                spo2_dl = float(out[1])

        st.metric("Device", last_device)
        st.metric("HR (classical)", f"{hr_bpm if hr_bpm is not None else '--'} bpm")
        st.metric("SpO2 (classical)", f"{spo2_classical if spo2_classical is not None else '--'} %")
        st.metric("HR (DL)", f"{hr_dl if hr_dl is not None else '--'} bpm")
        st.metric("SpO2 (DL)", f"{spo2_dl if spo2_dl is not None else '--'} %")
        st.write(f"Peaks detected: {len(peaks)}")

# Recent table and download
st.subheader("Recent raw samples (last 200 rows)")
st.dataframe(st.session_state.df.tail(200).reset_index(drop=True))
csv = st.session_state.df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV (recent samples)", csv, "ppg_samples.csv", "text/csv")

st.markdown("---")
st.markdown("Train a model with `train_dl.py` to enable DL predictions. DL model must be saved at `model.pt` in the same folder as this app.")
