# train_dl.py
"""
Train the PPG1DCNN on synthetic PPG windows.

Usage:
    python train_dl.py --epochs 30 --batch 64 --save-path model.pt
"""
import argparse
import math, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
from model import PPG1DCNN

# --- Synthetic PPG generator for a window ---
def make_ppg_window(length_s=8.0, fs=17, hr=72.0, spo2=98.0, noise=1.0, motion=0.2):
    L = int(length_s * fs)
    t = np.arange(L) / fs
    f = hr / 60.0
    ac_ir = 0.03 * np.sin(2*np.pi*f*t) + 0.005 * np.sin(2*np.pi*2*f*t)
    ac_red = 0.028 * np.sin(2*np.pi*f*t + 0.2) + 0.004 * np.sin(2*np.pi*2*f*t + 0.25)
    motion_component = motion * (0.02 * np.sin(2*np.pi*0.5*t) + 0.015 * np.sin(2*np.pi*0.25*t))
    noise_ir = noise * (np.random.rand(L) - 0.5) * 0.01
    noise_red = noise * (np.random.rand(L) - 0.5) * 0.01
    dc_ir = 1.0 + (np.random.rand()*0.02 - 0.01)
    dc_red = 0.95 + (np.random.rand()*0.02 - 0.01)
    ir = dc_ir + ac_ir + motion_component + noise_ir
    red = dc_red + ac_red + motion_component*0.9 + noise_red
    # Normalize per-channel (simple)
    ir = (ir - ir.mean()) / (ir.std() + 1e-8)
    red = (red - red.mean()) / (red.std() + 1e-8)
    return np.stack([ir, red], axis=0).astype(np.float32), np.array([hr, spo2], dtype=np.float32)

class SyntheticPPGDataset(Dataset):
    def __init__(self, n_samples=5000, window_s=8.0, fs=17):
        self.n = n_samples
        self.window_s = window_s
        self.fs = fs

    def __len__(self): return self.n

    def __getitem__(self, idx):
        # sample random physiological params
        hr = random.uniform(45, 110)            # bpm
        spo2 = random.uniform(80, 100)         # %
        noise = random.uniform(0.2, 2.0)
        motion = random.uniform(0.0, 0.6)
        x, y = make_ppg_window(self.window_s, self.fs, hr, spo2, noise, motion)
        return x, y

def collate_fn(batch):
    xs = np.stack([b[0] for b in batch], axis=0)
    ys = np.stack([b[1] for b in batch], axis=0)
    return torch.tensor(xs), torch.tensor(ys)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train = SyntheticPPGDataset(n_samples=args.n_train, window_s=args.window_s, fs=args.fs)
    ds_val = SyntheticPPGDataset(n_samples=args.n_val, window_s=args.window_s, fs=args.fs)
    loader = DataLoader(ds_train, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    vloader = DataLoader(ds_val, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    model = PPG1DCNN(in_channels=2).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val = 1e9
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(ds_train)
        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in vloader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
        val_loss = val_loss / len(ds_val)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  time={time.time()-t0:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  Saved model -> {args.save_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--n-train", type=int, dest="n_train", default=4000)
    p.add_argument("--n-val", type=int, dest="n_val", default=800)
    p.add_argument("--save-path", type=str, default="model.pt")
    p.add_argument("--window-s", type=float, default=8.0)
    p.add_argument("--fs", type=int, default=17)
    args = p.parse_args()
    train(args)
