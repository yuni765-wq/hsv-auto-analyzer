# -*- coding: utf-8 -*-
"""Thin hooks connecting Simulation ↔ ACE.
- simulate_sequence(): generate a short synthetic loop by latent traversal.
- run_ace_metrics(): adapter that calls existing ACE metrics on frames.
"""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .vae2d import VAE2D, VAEConfig


MODEL_PATH = Path('artifacts/sim/vae2d.pth')


# --- Utilities -----------------------------------------------------------------
def _load_model(device='cpu'):
ckpt = torch.load(MODEL_PATH, map_location=device)
z = ckpt.get('z_dim', 16)
model = VAE2D(VAEConfig(z_dim=z)).to(device)
model.load_state_dict(ckpt['state_dict'])
model.eval()
return model


def _to_torch(arr):
t = torch.from_numpy(arr[None, None, ...].astype('float32'))
return t


# --- Simulation ----------------------------------------------------------------
def simulate_sequence(seconds=1.0, fps=30, amp=2.0, freq_hz=3.0, base_img=None, device='cpu'):
"""Return ndarray [T, H, W] in [0,1].
If base_img (PIL/numpy) is None, decode from a zero latent.
Latent traversal is along e1 axis as a placeholder for GlOVe.
"""
assert MODEL_PATH.exists(), "Train and save a VAE checkpoint first."
model = _load_model(device)


T = int(seconds * fps)
H = W = 128
# base latent
z = torch.zeros((1, model.z_dim), dtype=torch.float32, device=device)
# traversal axis (placeholder for GlOVe). For now e1.
axis = torch.zeros_like(z); axis[:, 0] = 1.0


frames = []
with torch.no_grad():
for t in range(T):
phase = 2*np.pi*freq_hz*(t/fps)
zt = z + torch.tensor(amp*np.sin(phase), device=device).view(1,1) * axis
x_hat = model.dec(zt).cpu().numpy()[0,0]
frames.append(x_hat)
return np.stack(frames, axis=0)


# --- ACE Adapter ---------------------------------------------------------------


def run_ace_metrics(frames_np):
"""Adapter: expects frames_np [T,H,W] float in [0,1].
Convert to the format your ACE pipeline expects and call existing functions.
Replace the body with actual imports once wired into app.
"""
# TODO: integrate with existing ACE metrics (compute_envelope, detect_gat_..., etc.)
# Placeholder returns
return {
'GAT_ms': np.nan,
'GOT_ms': np.nan,
'VOnT_ms': np.nan,
'VOffT_ms': np.nan,
'AP': np.nan,
'TP': np.nan,
'QC': 'N/A',
}