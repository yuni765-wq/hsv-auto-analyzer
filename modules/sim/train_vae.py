# -*- coding: utf-8 -*-
import os, time
import torch
import streamlit as st
from pathlib import Path
from vae2d import VAE2D, VAEConfig, kl_divergence, recon_loss
from datasets import make_loaders


st.set_page_config(page_title='ACE – VAE Trainer', layout='wide')


ARTIFACT_DIR = Path('artifacts/sim'); ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / 'vae2d.pth'


col1, col2 = st.columns([2,1])
with col2:
z_dim = st.number_input('Latent dim (z)', 8, 128, 16, step=8)
lr = st.number_input('Learning rate', 1e-5, 1e-2, 1e-3, format='%e')
epochs = st.number_input('Epochs', 1, 500, 50)
batch = st.number_input('Batch size', 4, 128, 32)
beta_max = st.slider('KL β max', 0.0, 4.0, 1.0, 0.1)
warmup = st.slider('Warmup epochs', 0, 100, 30)
data_dir = st.text_input('Data base dir', 'data/hsv_frames')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write(f"Device: {device}")


with col1:
if st.button('Train VAE'):
train_dl, val_dl = make_loaders(data_dir, bsz=int(batch))
model = VAE2D(VAEConfig(z_dim=int(z_dim))).to(device)
opt = torch.optim.Adam(model.parameters(), lr=float(lr))


best_val = float('inf')
start = time.time()
for epoch in range(1, int(epochs)+1):
model.train(); total = 0.0
for x in train_dl:
x = x.to(device)
x_hat, mu, logvar = model(x)
beta = min(beta_max, epoch / max(1, warmup) * beta_max)
loss = recon_loss(x, x_hat) + beta * kl_divergence(mu, logvar)
opt.zero_grad(); loss.backward(); opt.step()
total += loss.item() * x.size(0)
train_loss = total / len(train_dl.dataset)


# quick val (L1 only for speed)
model.eval(); vtotal = 0.0
with torch.no_grad():
for x in val_dl:
x = x.to(device)
x_hat, mu, logvar = model(x)
vloss = recon_loss(x, x_hat)
vtotal += vloss.item() * x.size(0)
val_loss = vtotal / len(val_dl.dataset)
st.write(f"epoch {epoch:03d} train={train_loss:.4f} val(L1)={val_loss:.4f} beta={beta:.2f}")


# checkpoint
if val_loss < best_val:
best_val = val_loss
torch.save({'state_dict': model.state_dict(), 'z_dim': int(z_dim)}, MODEL_PATH)
st.success(f"Done. Best val(L1)={best_val:.4f} Saved → {MODEL_PATH} ({time.time()-start:.1f}s)")


if MODEL_PATH.exists():
st.info(f"Found model: {MODEL_PATH}")