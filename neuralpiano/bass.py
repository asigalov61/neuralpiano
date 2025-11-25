# Bass Python module

import math
from typing import Tuple, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

TensorLike = Union[torch.Tensor, np.ndarray]

def enhance_audio_bass(mono: TensorLike,
                       sr: int = 48000,
                       low_cutoff: float = 200.0,
                       ir_len: int = 1025,
                       low_gain_db: float = 8.0,
                       sub_mix: float = 0.75,
                       compressor_thresh_db: float = -24.0,
                       compressor_ratio: float = 3.0,
                       compressor_attack_ms: float = 8.0,
                       compressor_release_ms: float = 120.0,
                       makeup_db: float = 0.0,
                       drive: float = 1.15,
                       wet_mix: float = 0.9,
                       downsample_target: int = 4000,
                       device: Optional[torch.device] = None,
                       dtype: torch.dtype = torch.float32
                      ) -> Tuple[torch.Tensor, Dict]:
    
    """
    Fast bass enhancement optimized for GPU with robust shape alignment.

    Returns:
      enhanced 1-D torch.Tensor (same length as input), diagnostics dict.
    """
    
    # --- prepare input tensor ---
    if isinstance(mono, np.ndarray):
        x = torch.from_numpy(mono.astype(np.float32))
    elif isinstance(mono, torch.Tensor):
        x = mono.clone()
    else:
        raise TypeError("mono must be numpy or torch.Tensor")
    if x.ndim != 1:
        raise ValueError("mono must be 1-D (mono)")

    device = device or (x.device if isinstance(x, torch.Tensor) else torch.device('cpu'))
    x = x.to(device=device, dtype=dtype, copy=False)
    N = x.shape[-1]
    eps = 1e-12

    # --- helpers ---
    def db2lin(db): return 10.0 ** (db / 20.0)
    def next_pow2(n): return 1 << ((n - 1).bit_length())

    # linear-phase lowpass IR builder (small, efficient)
    def make_lowpass_ir(cutoff_hz, sr_local, length):
        if length % 2 == 0:
            length += 1
        t = torch.arange(length, device=device, dtype=dtype) - (length - 1) / 2.0
        sinc_arg = 2.0 * cutoff_hz / sr_local * t
        h = torch.sinc(sinc_arg)
        beta = 6.0
        # kaiser_window signature: (window_length, periodic=False, beta, *, dtype=None, layout=None, device=None)
        win = torch.kaiser_window(length, False, beta, dtype=dtype, device=device)
        h = h * win
        h = h / (h.sum() + eps)
        return h

    # FFT convolution using optional precomputed kernel FFT
    def fft_conv_signal_kernel(sig, kernel, kernel_fft=None):
        n = sig.shape[-1]
        k = kernel.shape[-1]
        out_len = n + k - 1
        size = next_pow2(out_len)
        S = torch.fft.rfft(sig, n=size)
        if kernel_fft is None:
            K = torch.fft.rfft(kernel, n=size)
        else:
            K = kernel_fft
        Y = S * K
        y = torch.fft.irfft(Y, n=size)[:out_len]
        return y[:n], K

    # --- 1) extract low band via FFT conv (single pass) ---
    lp = make_lowpass_ir(low_cutoff, sr, ir_len)
    low, lp_fft = fft_conv_signal_kernel(x, lp)  # low is same device/dtype

    # --- 2) downsample low band for fast processing ---
    ds = max(1, int(round(sr / float(downsample_target))))
    ds_sr = sr // ds
    if ds > 1:
        pad = (-low.shape[-1]) % ds
        if pad:
            low_p = F.pad(low.unsqueeze(0).unsqueeze(0), (0, pad)).squeeze(0).squeeze(0)
        else:
            low_p = low
        # decimate by averaging each block (cheap, preserves low-band energy)
        low_ds = low_p.view(-1, ds).mean(dim=1)
    else:
        low_ds = low

    # --- 3) compressor on downsampled low band (vectorized) ---
    win_len = max(1, int(round(0.01 * ds_sr)))
    sq = low_ds * low_ds
    pad = (win_len - 1) // 2
    sq_p = F.pad(sq.unsqueeze(0).unsqueeze(0), (pad, win_len - 1 - pad)).squeeze(0).squeeze(0)
    kernel = torch.ones(win_len, device=device, dtype=dtype) / float(win_len)
    rms_sq = F.conv1d(sq_p.unsqueeze(0).unsqueeze(0), kernel.view(1,1,-1)).squeeze(0).squeeze(0)
    rms = torch.sqrt(rms_sq + 1e-12)

    # soft-knee gain computer vectorized
    lvl_db = 20.0 * torch.log10(rms.clamp(min=1e-12))
    knee = 3.0
    over = lvl_db - compressor_thresh_db
    zero = torch.zeros_like(over)
    gr_db = torch.where(
        over <= -knee,
        zero,
        torch.where(
            over >= knee,
            compressor_thresh_db + (over / compressor_ratio) - lvl_db,
            - ((1.0 - 1.0/compressor_ratio) * (over + knee)**2) / (4.0 * knee)
        )
    ).clamp(max=0.0)
    gain_lin = 10.0 ** (gr_db / 20.0)

    # smooth gain with FIR approximations for attack/release
    def exp_kernel(tc_ms, sr_local, length=64):
        if length < 3:
            length = 3
        tau = max(1e-6, tc_ms / 1000.0)
        t = torch.arange(length, device=device, dtype=dtype)
        k = torch.exp(-t / (tau * sr_local))
        k = k / k.sum()
        return k

    atk_k = exp_kernel(compressor_attack_ms, ds_sr, length=64)
    rel_k = exp_kernel(compressor_release_ms, ds_sr, length=128)
    # convolve (padding may change length slightly)
    g_atk = F.conv1d(gain_lin.unsqueeze(0).unsqueeze(0), atk_k.view(1,1,-1), padding=(atk_k.numel()-1)//2).squeeze(0).squeeze(0)
    g_smooth = F.conv1d(g_atk.unsqueeze(0).unsqueeze(0), rel_k.view(1,1,-1), padding=(rel_k.numel()-1)//2).squeeze(0).squeeze(0)

    # --- ALIGN: ensure g_smooth matches low_ds length ---
    if g_smooth.shape[0] != low_ds.shape[0]:
        if g_smooth.shape[0] > low_ds.shape[0]:
            g_smooth = g_smooth[:low_ds.shape[0]]
        else:
            pad_len = low_ds.shape[0] - g_smooth.shape[0]
            if g_smooth.numel() == 0:
                # fallback to ones if something went wrong
                g_smooth = torch.ones(low_ds.shape[0], device=device, dtype=dtype)
            else:
                last = g_smooth[-1:].repeat(pad_len)
                g_smooth = torch.cat([g_smooth, last], dim=0)

    # apply makeup
    makeup_lin = db2lin(makeup_db)
    low_ds_comp = low_ds * g_smooth * makeup_lin

    # --- 4) subharmonic generation on downsampled signal ---
    rect = torch.clamp(low_ds_comp, min=0.0)
    lp_sub_len = 513 if ds_sr >= 4000 else 257
    lp_sub = make_lowpass_ir(200.0, ds_sr, lp_sub_len)
    rect_lp, _ = fft_conv_signal_kernel(rect, lp_sub)
    sub_gain = db2lin(low_gain_db)
    sub_ds = rect_lp * sub_gain

    # soft saturation (vectorized)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    sat_low_ds = torch.tanh(low_ds_comp * drive) / torch.tanh(one)
    sat_sub_ds = torch.tanh(sub_ds * (drive * 0.8)) / torch.tanh(one)
    enhanced_low_ds = (1.0 - sub_mix) * sat_low_ds + sub_mix * sat_sub_ds

    # --- 5) upsample enhanced low back to original rate ---
    if ds > 1:
        # ensure length before upsampling is consistent
        L_needed = (low.shape[-1] + ds - 1) // ds
        if enhanced_low_ds.shape[0] < L_needed:
            pad_len = L_needed - enhanced_low_ds.shape[0]
            enhanced_low_ds = torch.cat([enhanced_low_ds, enhanced_low_ds[-1:].repeat(pad_len)], dim=0)
        enhanced_low = F.interpolate(enhanced_low_ds.view(1,1,-1), scale_factor=ds, mode='linear', align_corners=False).view(-1)[:low.shape[-1]]
    else:
        enhanced_low = enhanced_low_ds
        # ensure exact length
        if enhanced_low.shape[0] != low.shape[-1]:
            if enhanced_low.shape[0] > low.shape[-1]:
                enhanced_low = enhanced_low[:low.shape[-1]]
            else:
                enhanced_low = torch.cat([enhanced_low, enhanced_low[-1:].repeat(low.shape[-1] - enhanced_low.shape[0])], dim=0)

    # --- 6) band-limit enhanced low using original lp FFT (cheap because we have lp_fft) ---
    enhanced_low_band, _ = fft_conv_signal_kernel(enhanced_low, lp, kernel_fft=lp_fft)

    # --- 7) scale wet and mix back ---
    wet = enhanced_low_band * db2lin(low_gain_db)
    out = (1.0 - wet_mix) * x + wet_mix * (x + wet)

    # final gentle limiter
    peak = float(out.abs().max().item())
    if peak > 0.999:
        out = out * (0.999 / peak)

    diagnostics = {
        "sr": sr,
        "low_cutoff": low_cutoff,
        "ir_len": ir_len,
        "low_gain_db": low_gain_db,
        "sub_mix": sub_mix,
        "downsample_factor": ds,
        "downsample_rate": ds_sr,
        "compressor_avg_gain_db": float(20.0 * math.log10(max(g_smooth.mean().item(), 1e-12))),
        "input_peak": float(x.abs().max().item()),
        "output_peak": float(out.abs().max().item()),
    }

    return out.to(dtype=dtype, device=device), diagnostics