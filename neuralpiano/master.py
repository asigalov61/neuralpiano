# Master Python module

import torch
import math
from typing import Dict, Tuple

Tensor = torch.Tensor

def master_mono_piano(mono: Tensor,
                      sr: int = 48000,
                      hp_cut: float = 20.0,
                      low_shelf_db: float = -1.5,
                      low_shelf_fc: float = 250.0,
                      presence_boost_db: float = 1.6,
                      presence_fc: float = 3500.0,
                      compressor_thresh_db: float = -6.0,
                      compressor_ratio: float = 2.5,
                      compressor_attack_ms: float = 8.0,
                      compressor_release_ms: float = 80.0,
                      compressor_makeup_db: float = 0.0,
                      stereo_spread_ms: float = 6.0,
                      stereo_spread_level_db: float = -12.0,
                      reverb_size_sec: float = 0.6,
                      reverb_mix: float = 0.06,
                      limiter_ceiling_db: float = -0.3,
                      dithering_bits: int = 24,
                      gain_db: float = 20.0,
                      device: torch.device = None,
                      dtype: torch.dtype = None,
                      iir_impulse_len: int = 2048,        # length used to approximate IIRs as FIRs (tradeoff speed/accuracy)
                    ) -> Tuple[Tensor, Dict]:
    """
    Mastering pipeline for a mono input piano track producing a stereo master and diagnostics.

    Description and design choices
    - Purpose: provide a compact, deterministic, and high-throughput mastering chain implemented
      with PyTorch tensors. The pipeline uses short FIR approximations for small IIR filters,
      single-shot FFT convolution for medium/long FIRs (cascaded HP + shelf, reverb), a fast
      RMS-based compressor, fractional-delay stereo widening, a smooth soft limiter, and
      deterministic TPDF dithering.
    - Where gain_db is applied: an explicit master gain parameter (gain_db) is applied after
      EQ/compression/stereo processing and reverb but before the soft limiter and final safety
      scaling. This placement allows the limiter to react to the applied gain, preserving
      consistent ceiling behavior while enabling transparent loudness adjustments and avoiding
      excessive post-limiter boosting which would bypass the limiter's protection.
    - Implementation details:
      * EQ: designs 2nd-order biquads and converts them to short FIR impulse responses using
        direct IIR stepping (iir_to_fir). HP and low-shelf are cascaded via FFT convolution.
      * Presence: implemented as a small symmetric FIR (sinc-windowed) and applied with conv1d.
      * Compression: fast RMS detector downsampled to ~4 kHz with an attack/release recurrence
        executed on CPU for determinism and efficiency. Soft-knee gain computer produces a
        time-varying linear gain applied to the signal with optional makeup gain.
      * Stereo widen: fractional sub-sample delays are used to create left/right channels, then
        mid/side processing adjusts side level.
      * Reverb: simple tapped + exponential-tail IR built and FFT-convolved once.
      * Limiter: soft tanh-based limiter that scales output to target ceiling (limiter_ceiling_db).
      * Dither: deterministic vectorized LCG generates TPDF dither for specified bit depth.
    - Determinism and diagnostics: the function avoids non-deterministic ops and returns a
      diagnostics dict with numeric metrics (peaks, RMS, average reduction, applied scales).
    - Precision and device handling: prefers float32 for performance; preserves device assignment;
      convolution routines use torch.fft (rfft/irfft) for CPU/GPU compatibility.

    Parameters
    - mono: Tensor of shape [N] or [1, N], mono PCM in float range roughly [-1, 1].
    - sr: sample rate in Hz.
    - hp_cut: high-pass cutoff frequency in Hz.
    - low_shelf_db: low-shelf gain (dB).
    - low_shelf_fc: low-shelf center freq (Hz).
    - presence_boost_db: narrow presence boost amount (dB).
    - presence_fc: center freq of presence boost (Hz).
    - compressor_...: compressor threshold (dBFS), ratio, attack and release (ms), and makeup (dB).
    - stereo_spread_ms: nominal stereo spread in milliseconds used to compute fractional delays.
    - stereo_spread_level_db: side gain in dB applied to M/S side channel.
    - reverb_size_sec: approximate reverb tail time in seconds (affects IR length).
    - reverb_mix: wet fraction of reverb to blend with dry signal.
    - limiter_ceiling_db: final soft limiter ceiling in dBFS (should be <= 0).
    - dithering_bits: integer bits for TPDF dithering (1-32). Use 0 or outside range to disable.
    - gain_db: master gain applied (dB). Positive increases loudness, negative reduces. Applied
      before the limiter so the limiter shapes peaks introduced by the gain.
    - device, dtype: optional torch.device and dtype to force placement/precision.
    - iir_impulse_len: length of FIR used to approximate 2nd-order IIRs (tradeoff accuracy/speed).

    Returns
    - stereo: Tensor shaped [2, N] float32 in range (-1, 1) representing left/right master.
    - diagnostics: Dict with numeric measurements and settings for reproducibility.

    Notes
    - The function keeps intermediary operations vectorized. Non-trivial recurrence for the RMS
      detector runs on CPU for stability and determinism but the returned envelopes are moved back
      to the target device and dtype.
    - For large inputs and GPU usage, FFT sizes are chosen as powers of two for efficiency.
    - If you want gain to be applied as a pre-EQ trim (instead of pre-limiter), move the gain
      multiplication earlier; current placement intentionally lets the limiter handle the gain.
    """

    # --- Setup, sanitize ---
    if mono.ndim == 2 and mono.shape[0] == 1:
        x = mono.squeeze(0)
    elif mono.ndim == 1:
        x = mono
    else:
        raise ValueError("mono must be shape [1, N] or [N]")

    device = device or x.device
    # prefer float32 for fastest throughput unless user explicitly provided float64
    if dtype is None:
        dtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    x = x.to(device=device, dtype=dtype, copy=False)
    N = x.shape[-1]
    eps = 1e-12

    diagnostics: Dict = {}
    def db2lin(d): return 10.0 ** (d / 20.0)
    def lin2db(v): return 20.0 * math.log10(max(v, 1e-12))

    # --- Helper: design biquad coefficients (as before) ---
    def design_butter_hp(fc, fs, Q=0.7071):
        omega = 2.0 * math.pi * fc / fs
        alpha = math.sin(omega) / (2.0 * Q)
        cosw = math.cos(omega)
        b0 =  (1 + cosw) / 2.0
        b1 = -(1 + cosw)
        b2 =  (1 + cosw) / 2.0
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha
        return b0, b1, b2, a0, a1, a2

    def design_low_shelf(fc, fs, gain_db, Q=0.7071):
        A = 10 ** (gain_db / 40.0)
        w0 = 2.0 * math.pi * fc / fs
        alpha = math.sin(w0) / (2.0 * Q)
        cosw = math.cos(w0)
        b0 =    A*( (A+1) - (A-1)*cosw + 2*math.sqrt(A)*alpha )
        b1 =  2*A*( (A-1) - (A+1)*cosw )
        b2 =    A*( (A+1) - (A-1)*cosw - 2*math.sqrt(A)*alpha )
        a0 =       (A+1) + (A-1)*cosw + 2*math.sqrt(A)*alpha
        a1 =  -2*( (A-1) + (A+1)*cosw )
        a2 =       (A+1) + (A-1)*cosw - 2*math.sqrt(A)*alpha
        return b0, b1, b2, a0, a1, a2

    # --- Utility: convert a 2nd-order IIR (b,a) to an FIR impulse response of length L
    # compute impulse response by stepping the IIR for L samples (this loop runs only for L ~ 1-4k)
    def iir_to_fir(b0, b1, b2, a0, a1, a2, L, device, dtype):
        # normalize
        b0n, b1n, b2n = b0 / a0, b1 / a0, b2 / a0
        a1n, a2n = a1 / a0, a2 / a0
        # compute impulse response on CPU or device (run on device if small L and device != cpu)
        run_device = device if (device.type != 'cuda' or L <= 8192) else device
        h = torch.zeros(L, device=run_device, dtype=dtype)
        x_prev1 = 0.0
        x_prev2 = 0.0
        y_prev1 = 0.0
        y_prev2 = 0.0
        # feed delta(0)=1, others 0
        for n in range(L):
            xv = 1.0 if n == 0 else 0.0
            yv = b0n * xv + b1n * x_prev1 + b2n * x_prev2 - a1n * y_prev1 - a2n * y_prev2
            h[n] = yv
            x_prev2 = x_prev1
            x_prev1 = xv
            y_prev2 = y_prev1
            y_prev1 = yv
        # ensure on main device
        if h.device != device:
            h = h.to(device=device, dtype=dtype)
        return h

    # --- 1) Build IIR -> FIR approximations for HP and shelf (single impulse responses) ---
    # keep impulse length small (configurable) to balance cost/accuracy
    b0,b1,b2,a0,a1,a2 = design_butter_hp(hp_cut, sr)
    hp_ir = iir_to_fir(b0,b1,b2,a0,a1,a2, iir_impulse_len, device, dtype)

    b0s,b1s,b2s,a0s,a1s,a2s = design_low_shelf(low_shelf_fc, sr, low_shelf_db)
    shelf_ir = iir_to_fir(b0s,b1s,b2s,a0s,a1s,a2s, iir_impulse_len, device, dtype)

    # cascade IIRs by convolving their IRs (use FFT convolution)
    def fft_convolve_full(sig, kernel, device, dtype):
        n = sig.shape[-1]
        k = kernel.shape[-1]
        out_len = n + k - 1
        size = 1 << ((out_len - 1).bit_length())
        # cast to complex-friendly dtype (float32/64)
        S = torch.fft.rfft(sig, n=size)
        K = torch.fft.rfft(kernel, n=size)
        Y = S * K
        y = torch.fft.irfft(Y, n=size)[:out_len]
        # return same length as input (valid-ish) by trimming convolution to center-left aligned (like original)
        return y[:n]

    # apply HP then shelf by convolving with cascaded IR (hp_ir * shelf_ir)
    casc_ir = fft_convolve_full(hp_ir, shelf_ir, device, dtype)[:max(hp_ir.numel(), shelf_ir.numel())]
    # normalize tiny numerical offsets
    casc_ir = casc_ir / (casc_ir.abs().sum().clamp(min=eps))

    # apply cascade IR to input (single FFT conv)
    x_hp_shelf = fft_convolve_full(x, casc_ir, device, dtype)

    # --- 2) Presence boost (small FIR) same approach but small kernel conv1d is very fast ---
    pres_len = min(256, max(65, int(sr * 0.0045)))  # ~4.5 ms
    t_idx = torch.arange(pres_len, device=device, dtype=dtype) - (pres_len - 1) / 2.0
    h = (torch.sinc(2.0 * presence_fc / sr * t_idx) * torch.hann_window(pres_len, device=device, dtype=dtype))
    h = h / (h.abs().sum() + eps)
    gain_lin = db2lin(presence_boost_db)
    presence_ir = (gain_lin - 1.0) * h
    presence_ir[(pres_len - 1) // 2] += 1.0
    # conv small kernel with conv1d (fast)
    x_eq = torch.nn.functional.conv1d(x_hp_shelf.view(1,1,-1), presence_ir.view(1,1,-1), padding=(pres_len-1)//2).view(-1)

    diagnostics.update({
        "hp_cut": hp_cut,
        "low_shelf_db": low_shelf_db,
        "presence_db": presence_boost_db,
        "presence_len": pres_len,
        "iir_impulse_len": iir_impulse_len,
    })

    # --- 3) Fast RMS compressor (vectorized with downsampled detector) ---
    sig = x_eq
    attack_tc = math.exp(-1.0 / max(1.0, (compressor_attack_ms * sr / 1000.0)))
    release_tc = math.exp(-1.0 / max(1.0, (compressor_release_ms * sr / 1000.0)))
    sq = sig * sig

    ds = max(1, int(sr // 4000))  # detector rate ~4kHz
    if ds > 1:
        # pad to multiple of ds
        pad = (-sq.shape[-1]) % ds
        if pad:
            sq_pad = torch.nn.functional.pad(sq, (0, pad))
        else:
            sq_pad = sq
        sq_ds = sq_pad.view(-1).reshape(-1, ds).mean(dim=1)
    else:
        sq_ds = sq

    # recurrence on small downsampled vector executed on CPU (cheap)
    sq_ds_cpu = sq_ds.detach().cpu()
    env_ds = torch.empty_like(sq_ds_cpu)
    s_val = float(sq_ds_cpu[0].item())
    a = attack_tc
    r = release_tc
    for i in range(sq_ds_cpu.shape[0]):
        v = float(sq_ds_cpu[i].item())
        if v > s_val:
            s_val = a * s_val + (1.0 - a) * v
        else:
            s_val = r * s_val + (1.0 - r) * v
        env_ds[i] = s_val
    env_ds = env_ds.to(device=device, dtype=dtype)

    if ds > 1:
        env = env_ds.repeat_interleave(ds)[:N]
    else:
        env = env_ds

    rms_env = torch.sqrt(torch.clamp(env, min=eps))
    lvl_db = 20.0 * torch.log10(torch.clamp(rms_env, min=1e-12))
    knee = 3.0
    over = lvl_db - compressor_thresh_db
    # soft knee
    zero = torch.zeros_like(over)
    gain_reduction_db = torch.where(
        over <= -knee,
        zero,
        torch.where(
            over >= knee,
            compressor_thresh_db + (over / compressor_ratio) - lvl_db,
            - ((1.0 - 1.0/compressor_ratio) * (over + knee)**2) / (4.0 * knee)
        )
    ).clamp(max=0.0)
    gain_lin = 10.0 ** (gain_reduction_db / 20.0)
    if ds > 1:
        gain_full = gain_lin.repeat_interleave(ds)[:N]
    else:
        gain_full = gain_lin
    makeup = db2lin(compressor_makeup_db)
    comp_out = sig * gain_full * makeup

    diagnostics.update({
        "compressor_thresh_db": compressor_thresh_db,
        "compressor_ratio": compressor_ratio,
        "compressor_attack_ms": compressor_attack_ms,
        "compressor_release_ms": compressor_release_ms,
        "compressor_makeup_db": compressor_makeup_db,
        "detector_downsample": ds,
        "avg_reduction_db": float((20.0 * torch.log10((gain_full.mean().clamp(min=1e-12))).item())),
    })

    # --- 4) Stereo widening with fractional sub-sample delays (vectorized) ---
    spread_samples = max(1e-4, stereo_spread_ms * sr / 1000.0)
    left_delay = spread_samples * 0.5
    right_delay = -spread_samples * 0.3333

    def fractional_delay_vec(sig, delay):
        n = sig.shape[-1]
        idx = torch.arange(n, device=device, dtype=dtype)
        pos = idx - delay
        pos_floor = pos.floor().long()
        pos_ceil = pos_floor + 1
        frac = (pos - pos_floor.to(dtype))
        pos_floor = pos_floor.clamp(0, n-1)
        pos_ceil = pos_ceil.clamp(0, n-1)
        s_floor = sig[pos_floor]
        s_ceil = sig[pos_ceil]
        return s_floor * (1.0 - frac) + s_ceil * frac

    left = 0.985 * comp_out + 0.015 * fractional_delay_vec(comp_out, left_delay)
    right = 0.985 * comp_out + 0.015 * fractional_delay_vec(comp_out, right_delay)

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    side = side * db2lin(stereo_spread_level_db)
    left = mid + side
    right = mid - side

    # --- 5) Reverb: build IR and FFT-convolve (single-shot) ---
    reverb_len = int(min(int(sr * reverb_size_sec), 65536))
    reverb_len = max(reverb_len, int(0.02 * sr))
    t = torch.arange(reverb_len, device=device, dtype=dtype)
    tail = torch.exp(-t / (reverb_size_sec * sr + 1e-12))
    taps_ms = [12, 23, 37, 53, 79]
    ir = torch.zeros(reverb_len, device=device, dtype=dtype)
    for i, tm in enumerate(taps_ms):
        idx = int(round(sr * tm / 1000.0))
        if idx < reverb_len:
            ir[idx] += (0.5 ** (i + 1))
    ir += 0.15 * tail
    ir = ir / (ir.abs().sum() + eps)

    left_rev = fft_convolve_full(left, ir, device, dtype)
    right_rev = fft_convolve_full(right, ir, device, dtype)
    left = (1.0 - reverb_mix) * left + reverb_mix * left_rev
    right = (1.0 - reverb_mix) * right + reverb_mix * right_rev

    diagnostics.update({
        "reverb_size_sec": reverb_size_sec,
        "reverb_mix": reverb_mix,
        "reverb_len": reverb_len,
    })

    # --- MASTER GAIN: apply desired gain in linear domain before limiter ---
    if abs(gain_db) > 1e-12:
        gain_lin_master = db2lin(gain_db)
        left = left * gain_lin_master
        right = right * gain_lin_master
        diagnostics['applied_gain_db'] = float(gain_db)
        diagnostics['applied_gain_lin'] = float(gain_lin_master)
    else:
        diagnostics['applied_gain_db'] = 0.0
        diagnostics['applied_gain_lin'] = 1.0

    # --- 6) Soft limiter ---
    def soft_limiter(x_chan, ceiling_db):
        ceiling_lin = db2lin(ceiling_db)
        peak = x_chan.abs().max().clamp(min=eps)
        if peak <= ceiling_lin:
            return x_chan
        scaled = x_chan * (ceiling_lin / peak)
        out = torch.tanh(scaled * 1.25) / 1.25
        out = out / out.abs().max().clamp(min=eps) * ceiling_lin
        return out

    left = soft_limiter(left, limiter_ceiling_db)
    right = soft_limiter(right, limiter_ceiling_db)

    # final safety scaling
    peak_val = max(left.abs().max().item(), right.abs().max().item())
    if peak_val > 0.999:
        scale = 0.999 / peak_val
        left = left * scale
        right = right * scale
        diagnostics['final_scale'] = float(scale)
    else:
        diagnostics['final_scale'] = 1.0

    # --- 7) Deterministic TPDF dithering (vectorized LCG) ---
    def vectorized_lcg(sz, seed):
        a = 1103515245
        c = 12345
        mod = 2**31
        seeds = (torch.arange(sz, device=device, dtype=torch.int64) * 1664525 + int(seed)) & (mod - 1)
        vals = (a * seeds + c) & (mod - 1)
        floats = (vals.to(dtype) / float(mod)) - 0.5
        return floats

    if 1 <= dithering_bits <= 32:
        q = 1.0 / (2 ** (dithering_bits - 1))
        seed = (N ^ sr ^ 0x9e3779b1) & 0xffffffff
        na = vectorized_lcg(N, seed)
        nb = vectorized_lcg(N, seed ^ 0x6a09e667)
        tpdf = (na - nb) * q
        left = left + 0.5 * tpdf
        right = right + 0.5 * tpdf

    # --- Output and diagnostics ---
    stereo = torch.stack([left.to(torch.float32), right.to(torch.float32)], dim=0)
    stereo = stereo.clamp(-1.0 + 1e-9, 1.0 - 1e-9)

    left_peak = left.abs().max().item()
    right_peak = right.abs().max().item()
    left_rms = math.sqrt(float(torch.mean(left * left).item()))
    right_rms = math.sqrt(float(torch.mean(right * right).item()))
    diagnostics.update({
        "left_peak": left_peak, "right_peak": right_peak,
        "left_peak_db": lin2db(left_peak), "right_peak_db": lin2db(right_peak),
        "left_rms_db": lin2db(left_rms), "right_rms_db": lin2db(right_rms),
        "num_samples": N, "sample_rate": sr,
    })

    return stereo, diagnostics