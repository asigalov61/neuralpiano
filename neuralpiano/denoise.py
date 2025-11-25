# Denoise Python module

import math
from typing import Tuple, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

TensorLike = Union[torch.Tensor, np.ndarray]

def denoise_audio(mono: TensorLike,
                  sr: int = 48000,
                  n_fft: int = 4096,
                  hop: Optional[int] = None,
                  noise_seconds: float = 0.8,
                  max_atten_db: float = 18.0,
                  noise_floor_db: float = -60.0,
                  smoothing_time_ms: float = 40.0,
                  smoothing_freq_bins: int = 3,
                  noise_sample: Optional[TensorLike] = None,
                  device: Optional[torch.device] = None,
                  dtype: torch.dtype = torch.float32
                 ) -> Tuple[torch.Tensor, Dict]:
    
    """
    Conservative denoiser tuned for solo piano.
    Returns denoised 1-D torch.Tensor and diagnostics dict.
    """
    
    # --- prepare tensor ---
    if isinstance(mono, np.ndarray):
        x = torch.from_numpy(mono.astype(np.float32))
    elif isinstance(mono, torch.Tensor):
        x = mono.clone()
    else:
        raise TypeError("mono must be a numpy array or torch.Tensor")

    if x.ndim != 1:
        raise ValueError("mono must be 1-D (mono)")

    device = device or (x.device if isinstance(x, torch.Tensor) else torch.device('cpu'))
    x = x.to(device=device, dtype=dtype, copy=False)
    N = x.shape[-1]
    eps = 1e-12

    hop = hop or (n_fft // 4)
    win = torch.hann_window(n_fft, device=device, dtype=dtype)

    # STFT / ISTFT helpers
    def stft_torch(sig):
        return torch.stft(sig, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=win, center=True, return_complex=True, pad_mode='reflect')

    def istft_torch(X, length):
        return torch.istft(X, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=win, center=True, length=length)

    # compute STFT
    X = stft_torch(x)  # complex tensor shape [freq_bins, frames]
    mag = X.abs() + eps
    freq_bins, frames = mag.shape

    # noise magnitude estimate
    if noise_sample is not None:
        if isinstance(noise_sample, np.ndarray):
            n_wav = torch.from_numpy(noise_sample.astype(np.float32)).to(device=device, dtype=dtype)
        else:
            n_wav = noise_sample.to(device=device, dtype=dtype)
        if n_wav.ndim > 1:
            n_wav = n_wav.mean(dim=1) if n_wav.shape[0] == 1 else n_wav.view(-1)
        Xn = stft_torch(n_wav)
        noise_mag = Xn.abs().mean(dim=1)
    else:
        if noise_seconds <= 0:
            frames_for_noise = 1
        else:
            approx_frames = max(1, int(math.ceil((noise_seconds * sr) / hop)))
            frames_for_noise = min(approx_frames, frames)
        noise_mag = mag[:, :frames_for_noise].mean(dim=1)

    # floor noise magnitude
    noise_floor_lin = 10.0 ** (noise_floor_db / 20.0)
    noise_mag = noise_mag.clamp(min=noise_floor_lin)

    # Wiener-like gain: S^2 / (S^2 + N^2)
    S2 = mag ** 2
    N2 = noise_mag.unsqueeze(1) ** 2
    G = S2 / (S2 + N2 + eps)  # shape [freq_bins, frames]

    # convert to attenuation dB and clamp
    att_db = -20.0 * torch.log10(G.clamp(min=1e-12))
    att_db_clamped = att_db.clamp(max=max_atten_db)
    G_limited = 10.0 ** (-att_db_clamped / 20.0)  # shape [freq_bins, frames]

    # -------------------------
    # Time smoothing (per-frequency)
    # -------------------------
    time_smooth_frames = max(1, int(round((smoothing_time_ms / 1000.0) * sr / hop)))
    if time_smooth_frames > 1:
        k = torch.hann_window(time_smooth_frames, device=device, dtype=dtype)
        k = k / k.sum()
        kernel = k.view(1, 1, -1).repeat(freq_bins, 1, 1)  # [freq_bins,1,k]
        inp = G_limited.unsqueeze(0)  # [1, freq_bins, frames]
        G_time = F.conv1d(inp, kernel, padding=(time_smooth_frames - 1) // 2, groups=freq_bins).squeeze(0)
    else:
        G_time = G_limited

    # -------------------------
    # Frequency smoothing (per-frame)
    # -------------------------
    if smoothing_freq_bins > 0:
        kf = torch.ones(smoothing_freq_bins, device=device, dtype=dtype)
        kf = kf / kf.sum()
        G_perm = G_time.permute(1, 0).unsqueeze(1)   # [frames, 1, freq_bins]
        kernel_f = kf.view(1, 1, -1)
        G_freq_perm = F.conv1d(G_perm, kernel_f, padding=(smoothing_freq_bins - 1) // 2).squeeze(1)  # [frames, freq_bins]
        G_freq = G_freq_perm.permute(1, 0)  # back to [freq_bins, frames]
    else:
        G_freq = G_time

    # Ensure orientation is [freq_bins, frames]
    if G_freq.ndim != 2:
        raise RuntimeError("Unexpected G_freq dimensionality")
    if G_freq.shape[0] != freq_bins and G_freq.shape[1] == freq_bins:
        G_freq = G_freq.permute(1, 0)

    # Build frequency-dependent protection vector [freq_bins, 1]
    freqs = torch.linspace(0.0, float(sr) / 2.0, steps=freq_bins, device=device, dtype=dtype).unsqueeze(1)
    low_protect = (freqs < 200.0).to(dtype)
    high_allow = (freqs > 6000.0).to(dtype)

    # apply frequency-dependent scaling (broadcast across frames)
    G_final = G_freq * (1.0 - 0.35 * low_protect) * (1.0 + 0.25 * high_allow)
    G_final = G_final.clamp(min=0.0, max=1.0)

    # -------------------------
    # ALIGNMENT FIX: ensure G_final matches X.shape exactly
    # -------------------------
    # If smoothing changed the frame count by ±1 (or more), trim or pad G_final to match X.
    # Trim if longer, pad by repeating last column if shorter.
    if G_final.shape[0] != freq_bins:
        # if frequency axis mismatched, try safe transpose or resize
        if G_final.shape[1] == freq_bins and G_final.shape[0] == frames:
            G_final = G_final.permute(1, 0)
        else:
            # fallback: resize frequency axis by trimming or repeating last row
            if G_final.shape[0] > freq_bins:
                G_final = G_final[:freq_bins, :]
            else:
                pad_rows = freq_bins - G_final.shape[0]
                last_row = G_final[-1:, :].repeat(pad_rows, 1)
                G_final = torch.cat([G_final, last_row], dim=0)

    if G_final.shape[1] != frames:
        if G_final.shape[1] > frames:
            G_final = G_final[:, :frames]
        else:
            # pad by repeating last column
            pad_cols = frames - G_final.shape[1]
            last_col = G_final[:, -1:].repeat(1, pad_cols)
            G_final = torch.cat([G_final, last_col], dim=1)

    # final safety check
    if G_final.shape != (freq_bins, frames):
        raise RuntimeError(f"Unable to align mask to STFT shape: mask {G_final.shape}, STFT {(freq_bins, frames)}")

    # apply mask (preserve phase)
    X_denoised = X * G_final

    # reconstruct
    y = istft_torch(X_denoised, length=N)

    # tiny residual low-frequency subtraction (gentle)
    lf_cut = 60.0
    lf_bin = int(round(lf_cut / (sr / 2.0) * (freq_bins - 1)))
    if lf_bin >= 1:
        lf_rms = mag[:lf_bin, :].mean().item()
        subtract = 0.02 * lf_rms
        y = y - subtract * torch.mean(y)

    # final safety normalize (very gentle)
    peak_in = float(x.abs().max().item())
    peak_out = float(y.abs().max().item())
    if peak_out > 0.999:
        y = y * (0.999 / peak_out)
        final_scale = 0.999 / peak_out
    else:
        final_scale = 1.0

    diagnostics = {
        "sr": sr,
        "n_fft": n_fft,
        "hop": hop,
        "noise_seconds": noise_seconds,
        "max_atten_db": max_atten_db,
        "smoothing_time_ms": smoothing_time_ms,
        "smoothing_freq_bins": smoothing_freq_bins,
        "input_peak": peak_in,
        "output_peak": float(y.abs().max().item()),
        "final_scale": final_scale,
    }

    return y.to(dtype=dtype, device=device), diagnostics