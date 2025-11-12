# streamlit_app_dl.py
"""
Streamlit dashboard + Flask receiver with disk-logging fallback.
The Flask receiver appends every received sample to incoming_log.csv.
Streamlit reads incoming_log.csv periodically and plots the samples.
"""

import streamlit as st
from threading import Thread
from queue import Queue, Empty
from flask import Flask, request, jsonify
import time, os, csv, pandas as pd, numpy as np, plotly.graph_objs as go

FLASK_PORT = 5000
LOG_CSV = "incoming_log.csv"
POLL_INTERVAL = 1.0  # seconds

# Ensure CSV exists with header
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["device_id", "t", "ir", "red"])

# ---------- Flask receiver (runs in background thread) ----------
def create_flask_app():
    app = Flask("receiver")

    @app.route("/", methods=["GET"])
    def root():
        return jsonify({"status":"receiver running", "post_endpoint":"/ingest"}), 200

    @app.route("/ingest", methods=["POST"])
    def ingest():
        # Append incoming samples to CSV (one row per sample)
        try:
            data = request.get_json(force=True)
        except Exception as e:
            return jsonify({"status":"error","reason":"invalid_json","detail":str(e)}), 200
        if not data or 'samples' not in data:
            return jsonify({"status":"error","reason":"no_samples"}), 200
        samples = data['samples']
        device_id = data.get('device_id', 'unknown')
        accepted = 0; dropped = 0
        rows = []
        for s in samples:
            try:
                ir_v = s.get('ir'); red_v = s.get('red')
                if ir_v is None or red_v is None:
                    dropped += 1
                    continue
                t_v = float(s.get('t', time.time()))
                rows.append([device_id, t_v, float(ir_v), float(red_v)])
                accepted += 1
            except Exception:
                dropped += 1
        # append to CSV atomically-ish (open, write, flush)
        if rows:
            try:
                with open(LOG_CSV, "a", newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(rows)
                    f.flush()
            except Exception as e:
                print("[receiver] failed to write CSV:", e)
                return jsonify({"status":"error","reason":"io_error","detail":str(e)}), 200
        print(f"[receiver] appended {accepted} rows (dropped={dropped})")
        return jsonify({"status":"ok","received":accepted,"dropped":dropped}), 200

    return app

def run_flask_in_thread():
    app = create_flask_app()
    def run():
        app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)
    t = Thread(target=run, daemon=True)
    t.start()
    return t

# ---------- Streamlit UI ----------
st.set_page_config(page_title="IoT PPG Dashboard (CSV fallback)", layout="wide")
st.title("IoT PPG Dashboard — CSV fallback receiver")
st.markdown(f"Receiver writes to `{LOG_CSV}` and Streamlit reads it every {POLL_INTERVAL}s. Flask port: {FLASK_PORT}")

if 'flask_started' not in st.session_state:
    st.session_state.flask_started = False
if not st.session_state.flask_started:
    run_flask_in_thread()
    st.session_state.flask_started = True
    st.success(f"Flask receiver running at http://localhost:{FLASK_PORT}/ingest")

# Controls
with st.sidebar:
    st.header("Controls")
    fs = st.slider("Assumed sample rate (Hz)", min_value=8, max_value=100, value=17)
    window_sec = st.slider("Window size (s) for display", min_value=2, max_value=600, value=8)
    min_peak_distance = st.slider("Min peak distance (ms)", min_value=200, max_value=1000, value=300)
    if st.button("Clear CSV log"):
        try:
            with open(LOG_CSV, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["device_id", "t", "ir", "red"])
            st.success("Cleared CSV log")
        except Exception as e:
            st.error("Failed to clear CSV: " + str(e))

# Read CSV (cached small-window reads are fine)
def read_log_csv(path):
    try:
        df = pd.read_csv(path)
        # ensure proper dtypes
        if not df.empty:
            df = df.dropna(subset=['t','ir','red'])
            df['t'] = df['t'].astype(float)
            df['ir'] = df['ir'].astype(float)
            df['red'] = df['red'].astype(float)
        return df
    except Exception as e:
        print("[streamlit] read CSV failed:", e)
        return pd.DataFrame(columns=['device_id','t','ir','red'])

# Poll every POLL_INTERVAL seconds (Streamlit re-runs on interaction; we force a short pause)
if 'last_poll' not in st.session_state:
    st.session_state.last_poll = 0.0

now = time.time()
if now - st.session_state.last_poll >= POLL_INTERVAL:
    # re-read CSV to update session DF
    df_all = read_log_csv(LOG_CSV)
    st.session_state.df = df_all.copy()
    st.session_state.last_poll = now
else:
    df_all = st.session_state.get('df', pd.DataFrame(columns=['device_id','t','ir','red']))

# Show status
col_s1, col_s2, col_s3 = st.columns([1,1,1])
with col_s1:
    st.metric("Rows in log file", int(len(df_all)))
with col_s2:
    try:
        size_kb = os.path.getsize(LOG_CSV) / 1024.0
        st.metric("Log file size (KB)", f"{size_kb:.1f}")
    except Exception:
        st.metric("Log file size (KB)", "--")
with col_s3:
    st.metric("Last read (unix)", int(st.session_state.last_poll))

# Plotting
cols = st.columns([3,1])
with cols[0]:
    st.subheader(f"Live PPG (last {window_sec}s)")
    if df_all.empty:
        st.write("No data yet — the receiver will append to the CSV when it receives POSTs.")
    else:
        latest_t = df_all['t'].max()
        window_df = df_all[df_all['t'] >= (latest_t - window_sec)]
        if window_df.empty:
            st.write("Waiting for more samples in the selected time window.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=window_df['t'], y=window_df['ir'], mode='lines', name='IR'))
            fig.add_trace(go.Scatter(x=window_df['t'], y=window_df['red'], mode='lines', name='Red'))
            fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=360, xaxis_title="time")
            st.plotly_chart(fig, use_container_width=True)

with cols[1]:
    st.subheader("Recent metrics (classical)")
    if df_all.empty:
        st.metric("HR (classical)", "-- bpm")
        st.metric("SpO2 (classical)", "-- %")
    else:
        last_device = df_all['device_id'].iloc[-1]
        device_df = df_all[df_all['device_id'] == last_device]
        latest_t = device_df['t'].max()
        window_df = device_df[device_df['t'] >= (latest_t - window_sec)]
        ir_vals = window_df['ir'].astype(float).to_numpy() if not window_df.empty else np.array([])
        red_vals = window_df['red'].astype(float).to_numpy() if not window_df.empty else np.array([])
        # simple peak-based HR
        def detect_peaks_simple(sig, min_distance_s=0.3, fs=17, threshold=None):
            sig = np.asarray(sig)
            if len(sig) < 3:
                return []
            if threshold is None:
                threshold = sig.mean() + 0.25 * sig.std()
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
        peaks = detect_peaks_simple(ir_vals, min_distance_s=min_peak_distance/1000.0, fs=fs)
        hr_bpm = None
        if len(peaks) >= 2:
            peak_times = window_df['t'].values[peaks]
            intervals = np.diff(peak_times)
            if len(intervals) > 0 and np.mean(intervals) > 0:
                hr_bpm = float(np.round(60.0 / np.mean(intervals), 1))
        def estimate_spo2_window(ir_vals, red_vals):
            if len(ir_vals) < 5 or len(red_vals) < 5:
                return None
            ac_ir = np.max(ir_vals) - np.min(ir_vals)
            ac_red = np.max(red_vals) - np.min(red_vals)
            dc_ir = np.mean(ir_vals)
            dc_red = np.mean(red_vals)
            if dc_ir == 0 or dc_red == 0 or ac_ir == 0:
                return None
            r = (ac_red / dc_red) / (ac_ir / dc_ir)
            spo2 = int(np.clip(int(round(110 - 25 * r)), 70, 100))
            return spo2
        spo2_classical = estimate_spo2_window(ir_vals, red_vals)
        st.metric("Device", last_device)
        st.metric("HR (classical)", f"{hr_bpm if hr_bpm is not None else '--'} bpm")
        st.metric("SpO2 (classical)", f"{spo2_classical if spo2_classical is not None else '--'} %")

st.subheader("Recent raw samples (tail 200)")
if df_all.empty:
    st.write("No samples in CSV yet.")
else:
    st.dataframe(df_all.tail(200).reset_index(drop=True))
