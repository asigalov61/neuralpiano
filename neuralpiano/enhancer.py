# Enhancer Python module

from typing import Union, Optional
import numpy as np
import torch
import torch.nn.functional as F
import math

TensorOrArray = Union[torch.Tensor, np.ndarray]

def _to_torch(x: TensorOrArray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = x.clone()
    if not torch.is_floating_point(t):
        t = t.float()
    return t.to(device=device, dtype=dtype).flatten()

def _to_output(t: torch.Tensor, orig: TensorOrArray, return_type: Optional[str]) -> TensorOrArray:
    out_type = return_type or ('numpy' if isinstance(orig, np.ndarray) else 'torch')
    if out_type == 'numpy':
        return t.cpu().numpy()
    return t

def _rms_val(x: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.sqrt(torch.mean(x**2) + eps).item())

def _soft_clip(x: torch.Tensor, drive: float = 1.0) -> torch.Tensor:
    return torch.tanh(x * drive) / (torch.tanh(torch.tensor(1.0, device=x.device)) + 1e-12)

def _multiband_compress(mag: torch.Tensor,
                        freq_bins: torch.Tensor,
                        sr: int,
                        bands: tuple = ((20,200),(200,2000),(2000,8000)),
                        thresholds_db: tuple = (-18.0, -18.0, -18.0),
                        ratios: tuple = (1.0, 1.8, 2.2),
                        attack_frames: int = 1,
                        release_frames: int = 8,
                        device: torch.device = torch.device('cpu'),
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    bins, frames = mag.shape
    out = mag.clone()
    for i, band in enumerate(bands):
        lo, hi = band
        mask = ((freq_bins >= lo) & (freq_bins < hi)).float().unsqueeze(1)
        if mask.sum() < 1:
            continue
        band_mag = (mag * mask).sum(dim=0) / (mask.sum() + 1e-12)
        if attack_frames > 0:
            env = torch.sqrt(F.conv1d(band_mag.unsqueeze(0).unsqueeze(0)**2,
                                     torch.ones(1,1,attack_frames, device=device, dtype=dtype)/attack_frames,
                                     padding=0).squeeze() + 1e-12)
        else:
            env = band_mag.abs()
        threshold = 10 ** (thresholds_db[i] / 20.0)
        ratio = ratios[i]
        gain = torch.ones_like(env)
        over = env > threshold
        if over.any():
            gain[over] = (threshold + (env[over] - threshold) / ratio) / (env[over] + 1e-12)
        kernel = torch.ones(release_frames, device=device, dtype=dtype) / float(release_frames)
        g = F.pad(gain.unsqueeze(0).unsqueeze(0), (release_frames//2, release_frames - 1 - release_frames//2), mode='replicate')
        g = F.conv1d(g, kernel.view(1,1,release_frames)).squeeze()
        out = out * (1.0 - mask) + (out * mask) * g.unsqueeze(0)
    return out

def enhance_audio_full(audio: TensorOrArray,
                       sr: int = 48000,
                       device: Union[str, torch.device] = 'cuda',
                       dtype: torch.dtype = torch.float32,
                       n_fft: int = 8192,
                       hop_length: int = 2048,
                       win_length: Optional[int] = None,
                       hp_cut_hz: float = 30.0,
                       denoise_strength: float = 0.55,
                       min_gain: float = 0.25,
                       time_smooth_k: int = 9,
                       freq_smooth_k: int = 15,
                       low_gain_db: float = -1.8,
                       mid_gain_db: float = 1.6,
                       high_gain_db: float = 1.8,
                       transient_boost: float = 1.12,
                       excite_amount: float = 0.01,
                       excite_scale: float = 0.02,
                       limiter_threshold_db: float = -0.5,
                       target_rms_db: float = -18.0,
                       return_type: Optional[str] = None,
                       verbose: bool = False
                      ) -> TensorOrArray:
    
    """
    Full-file STFT piano enhancement tuned to reduce reverb and increase clarity.
    This version fixes the residual smoothing length mismatch.
    """

    device = torch.device(device)
    x = _to_torch(audio, device=device, dtype=dtype)
    if x.dim() != 1:
        if x.dim() == 2:
            x = x.mean(dim=1)
        else:
            x = x.view(-1)
    n = x.numel()
    if win_length is None:
        win_length = n_fft

    if verbose:
        print(f"[enhance_v2_fixed] device={device}, dtype={dtype}, n={n}, n_fft={n_fft}, hop={hop_length}")

    window = torch.hann_window(win_length, device=device, dtype=dtype)

    # Full STFT
    X = torch.stft(x,
                   n_fft=n_fft,
                   hop_length=hop_length,
                   win_length=win_length,
                   window=window,
                   center=True,
                   return_complex=True)

    mag = torch.abs(X)
    phase = torch.angle(X)
    bins, frames = mag.shape

    freq_bins = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(device=device, dtype=dtype)
    hp_mask = (freq_bins >= hp_cut_hz).float().unsqueeze(1)
    low_mask = (freq_bins <= 200.0).float()
    mid_mask = ((freq_bins > 200.0) & (freq_bins <= 2000.0)).float()
    high_mask = (freq_bins > 2000.0).float()

    low_gain = 10 ** (low_gain_db / 20.0)
    mid_gain = 10 ** (mid_gain_db / 20.0)
    high_gain = 10 ** (high_gain_db / 20.0)
    band_gain = (low_gain * low_mask + mid_gain * mid_mask + high_gain * high_mask).unsqueeze(1)

    est_samples = min(int(0.5 * sr), n)
    est_frames = max(1, int(est_samples / hop_length))
    noise_floor = mag[:, :est_frames].median(dim=1).values.unsqueeze(1).clamp(min=1e-9)

    # Spectral subtraction + Wiener blend
    S2 = mag**2
    N2 = noise_floor**2
    over_sub = 1.0 + (denoise_strength * 0.6)
    sub = S2 - over_sub * N2
    sub = torch.clamp(sub, min=0.0)
    gain = sub / (S2 + 1e-12)
    gain = torch.clamp(gain, 0.0, 1.0)
    gain = 1.0 - (1.0 - gain) * denoise_strength

    # 2-D smoothing: time then frequency
    time_k = max(3, time_smooth_k if time_smooth_k % 2 == 1 else time_smooth_k + 1)
    freq_k = max(3, freq_smooth_k if freq_smooth_k % 2 == 1 else freq_smooth_k + 1)

    # Time smoothing (grouped conv across frames)
    gain_t = gain.unsqueeze(0)  # (1, bins, frames)
    bins_count = gain_t.shape[1]
    time_kernel = torch.ones(time_k, device=device, dtype=dtype) / float(time_k)
    k_time = time_kernel.view(1, 1, time_k).repeat(bins_count, 1, 1)
    pad_t = (time_k // 2, time_k - 1 - time_k // 2)
    gain_t = F.pad(gain_t, pad_t, mode='replicate')
    gain_t = F.conv1d(gain_t, k_time, groups=bins_count).squeeze(0)

    # Frequency smoothing (frames as batch)
    gain_f = gain_t.transpose(0, 1).unsqueeze(1)  # (frames,1,bins)
    freq_kernel = torch.ones(freq_k, device=device, dtype=dtype).view(1, 1, freq_k) / float(freq_k)
    pad_f = (freq_k // 2, freq_k - 1 - freq_k // 2)
    gain_f = F.pad(gain_f, pad_f, mode='replicate')
    gain_f = F.conv1d(gain_f, freq_kernel, groups=1)  # (frames,1,bins)
    gain = gain_f.squeeze(1).transpose(0, 1)  # (bins, frames)

    # Apply highpass mask and band shaping, enforce min_gain floor
    gain = gain * hp_mask
    mag = mag * gain * band_gain
    mag = torch.clamp(mag, min=min_gain * 1e-6)

    # Transient preservation (high-frequency energy rise)
    hf = (mag * high_mask.unsqueeze(1)).sum(dim=0)
    prev = F.pad(hf, (1, 0))[:-1]
    rise = torch.clamp((hf - prev) / (prev + 1e-9), min=0.0)
    transient_gain = 1.0 + (transient_boost - 1.0) * torch.clamp(rise * 2.0, 0.0, 1.0)
    mag = mag * transient_gain.unsqueeze(0)

    # Mild multiband compression
    mag = _multiband_compress(mag, freq_bins, sr,
                              bands=((20,200),(200,2000),(2000,8000)),
                              thresholds_db=(-22.0, -20.0, -20.0),
                              ratios=(1.0, 1.6, 1.8),
                              attack_frames=1,
                              release_frames=6,
                              device=device,
                              dtype=dtype)

    # Reconstruct complex STFT
    X = mag * torch.exp(1j * phase)

    # Very subtle harmonic excitation on highest band
    high_mask_full = (freq_bins > 4000.0).float().unsqueeze(1)
    X_high = X * high_mask_full
    high_time = torch.istft(X_high, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=n)
    excite = _soft_clip(high_time, drive=1.0 + excite_amount) - high_time
    E = torch.stft(excite, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, return_complex=True)
    X = X + excite_scale * E

    # ISTFT back to time domain
    out = torch.istft(X, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=n)

    # Final gentle de-reverb: smooth residual and subtract tiny fraction
    residual = out - x
    res = residual.cpu()
    kernel_len = max(1, int(0.05 * sr))  # 50 ms smoothing
    if kernel_len > 1 and res.numel() > kernel_len:
        # prepare kernel on CPU
        k = torch.ones(1, 1, kernel_len, dtype=res.dtype) / float(kernel_len)
        # conv1d expects (batch, channels, length)
        res_padded = F.pad(res.unsqueeze(0).unsqueeze(0), (kernel_len//2, kernel_len - 1 - kernel_len//2), mode='replicate')
        res_smooth = F.conv1d(res_padded, k, padding=0).squeeze()
        # Align length robustly: trim or pad to match n
        if res_smooth.numel() > n:
            res_smooth = res_smooth[:n]
        elif res_smooth.numel() < n:
            res_smooth = F.pad(res_smooth, (0, n - res_smooth.numel()))
        # subtract a tiny fraction of smoothed residual to reduce perceived reverb
        out = out - 0.02 * res_smooth.to(device=device, dtype=dtype)

    # Final limiter and RMS normalization
    peak = out.abs().max().clamp(min=1e-12).item()
    threshold = 10 ** (limiter_threshold_db / 20.0)
    if peak > threshold:
        out = out * (threshold / peak)

    current_rms = _rms_val(out)
    target_rms = 10 ** (target_rms_db / 20.0)
    if current_rms > 1e-12:
        out = out * (target_rms / current_rms)

    return _to_output(out, audio, return_type), audio.shape