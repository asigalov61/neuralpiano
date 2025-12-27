# enhancer.py

"""
A compact, dependency-light audio enhancement module implemented with PyTorch and NumPy.
This module provides a single high-level entry point `enhance_audio_full` that performs
STFT-domain denoising, smoothing, band shaping, transient preservation, mild multiband
compression, harmonic excitation, de-reverb (residual smoothing), limiting and RMS
normalization. Several internal helper utilities support device/dtype-safe conversions,
chunked smoothing for very long signals, and a simple multiband compressor applied to
magnitude spectrograms.

Design goals
------------
- **Simplicity**: Use only PyTorch and NumPy primitives (and optional tqdm for progress).
- **Device-aware**: All heavy computations are performed on the provided `device` and
  with the provided `dtype` to avoid unnecessary host/device transfers.
- **Memory-conscious**: Chunked smoothing and careful temporary deletion reduce peak
  memory usage; `torch.cuda.empty_cache()` calls are used opportunistically.
- **Practical defaults**: Reasonable default parameters for speech/music enhancement at
  common sampling rates (e.g., 48 kHz) while exposing many knobs for advanced users.

Primary function
----------------
- `enhance_audio_full(audio, sr=48000, device='cuda', dtype=torch.float32, ...)`
  Enhances a single-channel (mono) or multi-channel audio buffer and returns the
  processed audio (optionally duplicated to stereo) and the final output shape.

Notes and caveats
-----------------
- This module expects PyTorch to be installed and available. If CUDA is requested but
  not available, PyTorch will raise an error when creating the device.
- The implementation uses `torch.stft`/`torch.istft` with `return_complex=True` and
  reconstructs the signal using the original phase (with some harmonic excitation
  added). For best results, choose `n_fft`, `hop_length`, and `win_length` consistent
  with the signal characteristics and desired time-frequency resolution.
- The multiband compressor and smoothing kernels are intentionally simple and not a
  substitute for a production-grade dynamics processor; they are tuned for subtlety.
- The module performs in-place-like operations and frees temporaries to reduce memory
  pressure; however, users should still be mindful of GPU memory limits for large
  `n_fft` and long audio buffers.
- The function returns either a NumPy array or a PyTorch tensor depending on the
  `return_type` argument or the input type. When `output_as_stereo=True`, the returned
  tensor/array has shape `(2, n_samples)`; otherwise it is `(n_samples,)`.

Example
-------
>>> import numpy as np
>>> from enhancer import enhance_audio_full
>>> audio = np.random.randn(48000 * 5).astype(np.float32)  # 5 seconds of noise
>>> processed, shape = enhance_audio_full(audio, sr=48000, device='cpu', verbose=True, return_type='numpy')
>>> processed.shape, shape
((48000*5,), (240000,))  # example shapes (actual numbers depend on input length)
"""

from typing import Union, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# Optional progress bar
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

TensorOrArray = Union[torch.Tensor, np.ndarray]


def _maybe_tqdm(iterable, desc: str = "", verbose: bool = False):
    """
    Optionally wrap an iterable with a tqdm progress bar.

    Parameters
    ----------
    iterable : Iterable
        Any Python iterable (e.g., list, range) to iterate over.
    desc : str, optional
        Short description shown by tqdm when `verbose` is True.
    verbose : bool, optional
        If True and tqdm is available, return `tqdm(iterable, desc=desc)`.
        If tqdm is not available but `verbose` is True, print a simple progress
        message and return the original iterable. If False, return the original
        iterable without any side effects.

    Returns
    -------
    Iterable
        Either the original iterable or a tqdm-wrapped iterable when requested.
    """
    if not verbose:
        return iterable
    if _HAS_TQDM:
        return tqdm(iterable, desc=desc)
    print(f"[progress] {desc} ...")
    return iterable


def _to_torch(x: TensorOrArray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a NumPy array or a PyTorch tensor to a 1-D PyTorch tensor on the
    specified device and dtype.

    This helper flattens the result to a 1-D tensor because the processing
    pipeline in this module expects a single contiguous audio vector.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Input audio buffer. If a PyTorch tensor is provided it will be cloned
        to avoid modifying the caller's tensor.
    device : torch.device
        Target device for the returned tensor (e.g., `torch.device('cuda')`).
    dtype : torch.dtype
        Target dtype for the returned tensor (e.g., `torch.float32`).

    Returns
    -------
    torch.Tensor
        1-D tensor on `device` with `dtype`.
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = x.clone()
    if not torch.is_floating_point(t):
        t = t.float()
    return t.to(device=device, dtype=dtype).flatten()


def _to_output(t: torch.Tensor, orig: TensorOrArray, return_type: Optional[str]) -> TensorOrArray:
    """
    Convert a PyTorch tensor result back to the requested output type.

    Parameters
    ----------
    t : torch.Tensor
        Tensor to convert. Expected to be on CPU or GPU; if conversion to NumPy
        is requested, the tensor will be moved to CPU first.
    orig : numpy.ndarray or torch.Tensor
        The original input provided by the user. If `return_type` is None, this
        determines the default output type: return NumPy if `orig` was NumPy,
        otherwise return a PyTorch tensor.
    return_type : str or None
        Explicit output type requested by the caller. Supported values:
        - 'numpy' : return a NumPy ndarray
        - 'torch' : return a PyTorch tensor
        - None : infer from `orig` (NumPy -> numpy, else -> torch)

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Converted output in the requested format.
    """
    out_type = return_type or ('numpy' if isinstance(orig, np.ndarray) else 'torch')
    if out_type == 'numpy':
        return t.cpu().numpy()
    return t


def _rms_val(x: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute the root-mean-square (RMS) value of a tensor and return as float.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (time-domain signal).
    eps : float, optional
        Small epsilon added inside the square-root to avoid numerical issues.

    Returns
    -------
    float
        RMS value of `x`.
    """
    return float(torch.sqrt(torch.mean(x**2) + eps).item())


def _soft_clip(x: torch.Tensor, drive: float = 1.0) -> torch.Tensor:
    """
    Apply a gentle soft-clipping nonlinearity to a time-domain signal.

    The function uses a scaled hyperbolic tangent to provide a smooth saturation
    curve. This is used for subtle harmonic excitation generation.

    Parameters
    ----------
    x : torch.Tensor
        Input time-domain signal.
    drive : float, optional
        Drive factor controlling the amount of saturation. Values > 1 increase
        the nonlinearity strength.

    Returns
    -------
    torch.Tensor
        Soft-clipped signal with the same shape and device/dtype as `x`.
    """
    one = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return torch.tanh(x * drive) / (torch.tanh(one) + 1e-12)


def _multiband_compress(mag: torch.Tensor,
                        freq_bins: torch.Tensor,
                        sr: int,
                        bands: tuple = ((20, 200), (200, 2000), (2000, 8000)),
                        thresholds_db: tuple = (-18.0, -18.0, -18.0),
                        ratios: tuple = (1.0, 1.8, 2.2),
                        attack_frames: int = 1,
                        release_frames: int = 8,
                        device: torch.device = torch.device('cpu'),
                        dtype: torch.dtype = torch.float32,
                        verbose: bool = False) -> torch.Tensor:
    """
    Apply a simple multiband compressor to a magnitude spectrogram.

    The compressor operates on the magnitude spectrogram `mag` (shape: bins x frames)
    by computing a per-band envelope (mean magnitude across the band's frequency bins),
    applying a threshold/ratio rule to compute a gain curve, smoothing the gain with a
    release kernel, and applying the resulting gain to the magnitude bins belonging to
    that band.

    This is intentionally a lightweight, non-real-time-friendly implementation meant
    for offline processing of full spectrograms.

    Parameters
    ----------
    mag : torch.Tensor
        Magnitude spectrogram of shape (bins, frames).
    freq_bins : torch.Tensor
        Frequency values (Hz) for each bin (length == bins). Should be on the same
        device/dtype as `mag` or will be moved to `mag`'s device/dtype internally.
    sr : int
        Sampling rate in Hz (used for band definitions if needed).
    bands : tuple of (lo, hi) tuples, optional
        Frequency ranges (inclusive lower bound, exclusive upper bound) for each band.
    thresholds_db : tuple of floats, optional
        Thresholds for each band in decibels (dB). Converted to linear amplitude.
    ratios : tuple of floats, optional
        Compression ratios for each band. A ratio of 1.0 means no compression.
    attack_frames : int, optional
        Number of frames used to compute a short attack envelope (via convolution).
        If <= 0, the instantaneous magnitude is used.
    release_frames : int, optional
        Number of frames used to smooth the computed gain (moving average). If <= 0,
        no release smoothing is applied.
    device : torch.device, optional
        Device to use for temporary kernels and computations (defaults to CPU).
    dtype : torch.dtype, optional
        Dtype to use for temporary kernels and computations.
    verbose : bool, optional
        If True, show progress via `_maybe_tqdm`.

    Returns
    -------
    torch.Tensor
        Compressed magnitude spectrogram with the same shape and device/dtype as `mag`.

    Implementation notes
    --------------------
    - The per-band envelope is computed as the mean magnitude across the band's bins.
    - Thresholds are converted from dB to linear amplitude using `10^(dB/20)`.
    - Gain for frames above threshold is computed as:
        gain_over = (threshold + (env - threshold) / ratio) / env
      which reduces level according to the ratio while preserving phase later.
    - Release smoothing is implemented as a 1-D convolution with a uniform kernel.
    - The function multiplies the original `mag` by the computed per-band gain for
      the bins in that band and leaves other bins unchanged.
    """
    bins, frames = mag.shape
    out = mag.clone()

    # Ensure freq_bins on same device/dtype as mag for comparisons
    freq_bins = freq_bins.to(device=mag.device, dtype=mag.dtype)

    for i, band in enumerate(_maybe_tqdm(bands, desc="multiband compress", verbose=verbose)):
        lo, hi = band
        mask_bool = (freq_bins >= float(lo)) & (freq_bins < float(hi))
        if mask_bool.sum() < 1:
            continue
        mask = mask_bool.to(dtype=mag.dtype).unsqueeze(1)  # (bins,1)

        denom = mask.sum() + 1e-12
        band_mag = (mag * mask).sum(dim=0) / denom  # (frames,)

        # keep band_mag on same device/dtype as mag
        band_mag = band_mag.to(device=mag.device, dtype=mag.dtype)

        kernel_dtype = band_mag.dtype
        kernel_device = band_mag.device

        if attack_frames > 0:
            attack_kernel = torch.ones(1, 1, attack_frames, device=kernel_device, dtype=kernel_dtype) / float(attack_frames)
            sq = band_mag.unsqueeze(0).unsqueeze(0) ** 2
            env_conv = F.conv1d(sq, attack_kernel, padding=0)
            env = torch.sqrt(env_conv.squeeze() + 1e-12)
            del sq, env_conv
        else:
            env = band_mag.abs()

        threshold = 10 ** (thresholds_db[i] / 20.0)
        ratio = ratios[i]
        gain = torch.ones_like(env, dtype=kernel_dtype, device=kernel_device)
        over = env > threshold
        if over.any():
            gain_over = (threshold + (env[over] - threshold) / ratio) / (env[over] + 1e-12)
            gain = gain.clone()
            gain[over] = gain_over

        if release_frames > 0:
            release_kernel = torch.ones(release_frames, device=kernel_device, dtype=kernel_dtype) / float(release_frames)
            pad_left = release_frames // 2
            pad_right = release_frames - 1 - pad_left
            g = F.pad(gain.unsqueeze(0).unsqueeze(0), (pad_left, pad_right), mode='replicate')
            g = F.conv1d(g, release_kernel.view(1, 1, release_frames)).squeeze()
        else:
            g = gain

        out = out * (1.0 - mask) + (out * mask) * g.unsqueeze(0)

        # free temporaries
        del mask, band_mag, gain, g
        torch.cuda.empty_cache()

    return out


def _smooth_1d_chunked(signal: torch.Tensor,
                       kernel_len: int,
                       device: torch.device,
                       dtype: torch.dtype,
                       chunk_size: int = 10_000_000,
                       verbose: bool = False) -> torch.Tensor:
    """
    Compute a moving-average smoothing of a 1-D signal using chunked overlap processing.

    This function is intended for very long signals where allocating a full-length
    convolution result on the CPU or GPU would be memory-prohibitive. It processes
    the signal in chunks, padding each chunk by the kernel half-width so that the
    convolution output aligns with the original indices.

    Parameters
    ----------
    signal : torch.Tensor
        1-D tensor containing the signal to smooth. Must be on the target device.
    kernel_len : int
        Length of the moving-average kernel in samples. If <= 1, the function returns
        a clone of `signal`.
    device : torch.device
        Device to perform convolution on (should match `signal.device`).
    dtype : torch.dtype
        Dtype for the convolution kernel and intermediate results.
    chunk_size : int, optional
        Maximum number of output samples processed per chunk. Larger values reduce
        overhead but increase peak memory usage.
    verbose : bool, optional
        If True, show progress via `_maybe_tqdm`.

    Returns
    -------
    torch.Tensor
        Smoothed 1-D tensor on the same device/dtype as requested (dtype/device).
    """
    if kernel_len <= 1:
        return signal.clone()

    L = signal.numel()
    pad_left = kernel_len // 2
    pad_right = kernel_len - 1 - pad_left
    # kernel on device/dtype
    kernel = torch.ones(kernel_len, device=device, dtype=dtype) / float(kernel_len)
    kernel = kernel.view(1, 1, kernel_len)

    out = torch.empty_like(signal, device=device, dtype=dtype)

    # Determine chunk boundaries (process output indices)
    starts = list(range(0, L, chunk_size))
    iterator = _maybe_tqdm(starts, desc="residual smoothing chunks", verbose=verbose)

    for s in iterator:
        e = min(s + chunk_size, L)
        # slice input with overlap region to cover kernel support
        in_s = max(0, s - pad_left)
        in_e = min(L, e + pad_right)
        seg = signal[in_s:in_e]  # on device

        # seg length may be small; we will pad seg so conv output length equals seg length
        seg_b = seg.unsqueeze(0).unsqueeze(0)  # (1,1,seg_len)
        # pad seg_b by kernel half-widths so conv output length == seg_len
        seg_b = F.pad(seg_b, (pad_left, pad_right), mode='replicate')

        # ensure kernel dtype/device matches seg_b
        k = kernel.to(dtype=seg_b.dtype, device=seg_b.device)
        conv = F.conv1d(seg_b, k, padding=0).squeeze()  # length == seg_len

        # compute indices in conv result that correspond to output indices [s:e]
        conv_start = s - in_s
        conv_end = conv_start + (e - s)
        # assign into output
        out[s:e] = conv[conv_start:conv_end].to(dtype=dtype, device=device)

        # free temporaries
        del seg, seg_b, conv, k
        torch.cuda.empty_cache()

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
                       overall_gain_db: float = -1.0,
                       output_as_stereo: bool = False,
                       return_type: Optional[str] = None,
                       verbose: bool = False,
                       residual_chunk_size: int = 10_000_000
                       ) -> Tuple[TensorOrArray, Tuple[int, ...]]:
    """
    Enhance a full audio buffer using STFT-domain processing and return the processed
    audio together with the final output shape.

    The processing pipeline includes:
      1. STFT (magnitude/phase decomposition)
      2. Noise estimation from the initial frames and spectral subtraction
      3. Wiener-like blending and 2-D smoothing (time then frequency)
      4. High-pass masking and band-specific gain shaping
      5. Transient preservation via high-frequency rise detection
      6. Mild multiband compression applied to magnitude spectrogram
      7. Harmonic excitation derived from the highest frequency band
      8. ISTFT to reconstruct time-domain signal
      9. Residual smoothing (de-reverb) via chunked moving-average subtraction
     10. Overall gain, limiting, and RMS normalization
     11. Optional duplication to stereo with per-channel normalization

    Parameters
    ----------
    audio : numpy.ndarray or torch.Tensor
        Input audio buffer. Can be 1-D (mono) or 2-D (channels x samples or samples x channels).
        If a NumPy array is provided, the returned output will be NumPy by default unless
        `return_type` is specified.
    sr : int, optional
        Sampling rate in Hz. Used for frequency bin calculations and smoothing kernel lengths.
    device : str or torch.device, optional
        Device to perform computations on (e.g., 'cuda' or 'cpu'). Converted to `torch.device`.
    dtype : torch.dtype, optional
        Floating-point dtype used for processing (e.g., `torch.float32`).
    n_fft : int, optional
        FFT size used for STFT/ISTFT. Larger values increase frequency resolution at the
        cost of time resolution and memory.
    hop_length : int, optional
        Hop length (frame shift) in samples for STFT/ISTFT.
    win_length : int or None, optional
        Window length for STFT/ISTFT. If None, defaults to `n_fft`.
    hp_cut_hz : float, optional
        High-pass cutoff frequency (Hz). Frequencies below this are attenuated to reduce
        low-frequency noise and rumble.
    denoise_strength : float, optional
        Controls the aggressiveness of spectral subtraction and Wiener blending. Values
        closer to 1.0 increase denoising strength.
    min_gain : float, optional
        Minimum magnitude floor (linear) applied after spectral processing to avoid
        numerical issues and extremely low magnitudes.
    time_smooth_k : int, optional
        Kernel length (frames) for time-domain smoothing of the spectral gain. Odd values
        are preferred; the function will coerce to an odd value >= 3.
    freq_smooth_k : int, optional
        Kernel length (bins) for frequency-domain smoothing of the spectral gain. Odd
        values are preferred; the function will coerce to an odd value >= 3.
    low_gain_db, mid_gain_db, high_gain_db : float, optional
        Per-band gain adjustments in decibels applied after denoising. Converted to linear.
    transient_boost : float, optional
        Multiplier applied to detected high-frequency transients to preserve attack.
        Values > 1.0 increase transient emphasis.
    excite_amount : float, optional
        Drive parameter for the soft-clipper used to synthesize subtle harmonic content.
    excite_scale : float, optional
        Scaling factor applied to the STFT of the synthesized excitation before adding
        it back into the main STFT.
    limiter_threshold_db : float, optional
        Peak limiter threshold in dBFS. Values near 0 dB are louder; negative values
        reduce maximum peak amplitude.
    target_rms_db : float, optional
        Target RMS level in dBFS for final normalization.
    overall_gain_db : float, optional
        Global gain in dB applied before final limiting and normalization.
    output_as_stereo : bool, optional
        If True, duplicate the mono output to two channels and normalize each channel
        to the target RMS. If the input was multi-channel and you want to preserve
        channels, pre-process accordingly before calling this function.
    return_type : str or None, optional
        If 'numpy', return a NumPy array; if 'torch', return a PyTorch tensor; if None,
        infer from the input type (`audio`).
    verbose : bool, optional
        If True, print stage markers and progress information.
    residual_chunk_size : int, optional
        Chunk size (samples) used by `_smooth_1d_chunked` when smoothing the residual
        for de-reverb. Increase to reduce overhead for long signals, decrease to reduce
        peak memory usage.

    Returns
    -------
    processed_audio : numpy.ndarray or torch.Tensor
        Enhanced audio in the requested `return_type`. Shape is `(n_samples,)` for mono
        or `(2, n_samples)` when `output_as_stereo=True`.
    final_shape : tuple of ints
        Tuple describing the shape of the returned audio (e.g., `(n_samples,)` or `(2, n_samples)`).

    Warnings
    --------
    - This function is not real-time safe: it performs full-buffer STFT/ISTFT and
      several global operations that require access to the entire signal.
    - For very long audio buffers and large `n_fft`, GPU memory usage can be high.
      Consider processing in segments or using smaller FFT sizes if memory is limited.

    Implementation details
    ----------------------
    - Noise floor is estimated using the median magnitude across the first `est_frames`
      frames (where `est_frames` is derived from `sr` and `hop_length`).
    - Spectral subtraction uses a mild over-subtraction factor derived from `denoise_strength`.
    - Time and frequency smoothing are implemented as grouped 1-D convolutions for
      efficiency and to preserve per-bin independence where appropriate.
    - Multiband compression is applied to the magnitude spectrogram with three default
      bands tuned for low/mid/high content.
    - Harmonic excitation is synthesized by soft-clipping the time-domain signal
      reconstructed from the highest frequency band and adding a scaled STFT of that
      excitation back into the main STFT before ISTFT.
    """
    device = torch.device(device)
    x = _to_torch(audio, device=device, dtype=dtype)
    if x.dim() != 1:
        if x.dim() == 2:
            if x.shape[0] <= 2 and x.shape[0] < x.shape[1]:
                x = x.mean(dim=0)
            else:
                x = x.mean(dim=1)
        else:
            x = x.view(-1)
    n = x.numel()
    if win_length is None:
        win_length = n_fft

    if verbose:
        print(f"[neural piano enhancer] device={device}, dtype={dtype}, n={n}, n_fft={n_fft}, hop={hop_length}, overall_gain_db={overall_gain_db}, output_as_stereo={output_as_stereo}")

    # Window on device/dtype
    window = torch.hann_window(win_length, device=device, dtype=dtype)

    # STFT
    if verbose:
        print("[stage] STFT ...")
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

    # freq_bins on same dtype/device as mag
    freq_bins = torch.fft.rfftfreq(n_fft, 1.0 / sr).to(device=device, dtype=mag.dtype)

    hp_mask = (freq_bins >= hp_cut_hz).to(dtype=mag.dtype).unsqueeze(1)
    low_mask = (freq_bins <= 200.0).to(dtype=mag.dtype)
    mid_mask = ((freq_bins > 200.0) & (freq_bins <= 2000.0)).to(dtype=mag.dtype)
    high_mask = (freq_bins > 2000.0).to(dtype=mag.dtype)

    low_gain = 10 ** (low_gain_db / 20.0)
    mid_gain = 10 ** (mid_gain_db / 20.0)
    high_gain = 10 ** (high_gain_db / 20.0)
    band_gain = (low_gain * low_mask + mid_gain * mid_mask + high_gain * high_mask).unsqueeze(1).to(dtype=mag.dtype, device=device)

    est_samples = min(int(0.5 * sr), n)
    est_frames = max(1, int(est_samples / hop_length))
    noise_floor = mag[:, :est_frames].median(dim=1).values.unsqueeze(1).clamp(min=1e-9)

    # Spectral subtraction + Wiener blend
    if verbose:
        print("[stage] spectral subtraction and Wiener blend ...")
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
    if verbose:
        print("[stage] time smoothing ...")
    gain_t = gain.unsqueeze(0)  # (1, bins, frames)
    bins_count = gain_t.shape[1]
    time_kernel = torch.ones(time_k, device=device, dtype=mag.dtype) / float(time_k)
    k_time = time_kernel.view(1, 1, time_k).repeat(bins_count, 1, 1)
    pad_t = (time_k // 2, time_k - 1 - time_k // 2)
    gain_t = F.pad(gain_t, pad_t, mode='replicate')
    k_time = k_time.to(dtype=gain_t.dtype, device=gain_t.device)
    gain_t = F.conv1d(gain_t, k_time, groups=bins_count).squeeze(0)
    del k_time, time_kernel
    torch.cuda.empty_cache()

    # Frequency smoothing (frames as batch)
    if verbose:
        print("[stage] frequency smoothing ...")
    gain_f = gain_t.transpose(0, 1).unsqueeze(1)  # (frames,1,bins)
    freq_kernel = torch.ones(freq_k, device=device, dtype=mag.dtype).view(1, 1, freq_k) / float(freq_k)
    pad_f = (freq_k // 2, freq_k - 1 - freq_k // 2)
    gain_f = F.pad(gain_f, pad_f, mode='replicate')
    freq_kernel = freq_kernel.to(dtype=gain_f.dtype, device=gain_f.device)
    gain_f = F.conv1d(gain_f, freq_kernel, groups=1)  # (frames,1,bins)
    gain = gain_f.squeeze(1).transpose(0, 1)  # (bins, frames)
    del gain_f, freq_kernel
    torch.cuda.empty_cache()

    # Apply highpass mask and band shaping, enforce min_gain floor
    gain = gain * hp_mask
    mag = mag * gain * band_gain
    mag = torch.clamp(mag, min=min_gain * 1e-6)
    del gain, band_gain
    torch.cuda.empty_cache()

    # Transient preservation (high-frequency energy rise)
    if verbose:
        print("[stage] transient preservation ...")
    hf = (mag * high_mask.unsqueeze(1)).sum(dim=0)
    prev = F.pad(hf, (1, 0))[:-1]
    rise = torch.clamp((hf - prev) / (prev + 1e-9), min=0.0)
    transient_gain = 1.0 + (transient_boost - 1.0) * torch.clamp(rise * 2.0, 0.0, 1.0)
    mag = mag * transient_gain.unsqueeze(0)
    del hf, prev, rise, transient_gain
    torch.cuda.empty_cache()

    # Mild multiband compression
    if verbose:
        print("[stage] multiband compression ...")
    mag = _multiband_compress(mag, freq_bins, sr,
                              bands=((20, 200), (200, 2000), (2000, 8000)),
                              thresholds_db=(-22.0, -20.0, -20.0),
                              ratios=(1.0, 1.6, 1.8),
                              attack_frames=1,
                              release_frames=6,
                              device=device,
                              dtype=dtype,
                              verbose=verbose)
    torch.cuda.empty_cache()

    # Reconstruct complex STFT
    X = mag * torch.exp(1j * phase)
    del mag, phase
    torch.cuda.empty_cache()

    # Very subtle harmonic excitation on highest band
    if verbose:
        print("[stage] harmonic excitation ...")
    high_mask_full = (freq_bins > 4000.0).to(dtype=X.dtype).unsqueeze(1)
    X_high = X * high_mask_full
    high_time = torch.istft(X_high, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=n)
    del X_high, high_mask_full
    torch.cuda.empty_cache()

    excite = _soft_clip(high_time, drive=1.0 + excite_amount) - high_time
    E = torch.stft(excite, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, return_complex=True)
    X = X + excite_scale * E
    del E, excite, high_time
    torch.cuda.empty_cache()

    # ISTFT back to time domain
    if verbose:
        print("[stage] ISTFT ...")
    out = torch.istft(X, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=n)
    del X
    torch.cuda.empty_cache()

    # Final gentle de-reverb: smooth residual and subtract tiny fraction
    if verbose:
        print("[stage] residual smoothing (de-reverb) ...")
    residual = out - x  # keep on device
    # perform smoothing on device in chunks to avoid huge CPU allocations
    kernel_len = max(1, int(0.05 * sr))  # 50 ms smoothing
    if kernel_len > 1 and residual.numel() > kernel_len:
        # Use dtype/device consistent smoothing and chunking
        res_smooth = _smooth_1d_chunked(residual, kernel_len=kernel_len, device=device, dtype=dtype,
                                        chunk_size=residual_chunk_size, verbose=verbose)
        # subtract a tiny fraction of smoothed residual to reduce perceived reverb
        out = out - 0.02 * res_smooth
        del res_smooth
        torch.cuda.empty_cache()

    # Apply overall gain in dB (before final limiter/normalization)
    if abs(overall_gain_db) > 1e-6:
        if verbose:
            print(f"[stage] applying overall gain {overall_gain_db} dB ...")
        gain_lin = 10 ** (overall_gain_db / 20.0)
        out = out * gain_lin

    # Final limiter and RMS normalization
    if verbose:
        print("[stage] final limiter and RMS normalization ...")
    peak = out.abs().max().clamp(min=1e-12).item()
    threshold = 10 ** (limiter_threshold_db / 20.0)
    if peak > threshold:
        out = out * (threshold / peak)

    current_rms = _rms_val(out)
    target_rms = 10 ** (target_rms_db / 20.0)
    if current_rms > 1e-12:
        out = out * (target_rms / current_rms)

    # If stereo output requested: duplicate mono to two channels and normalize (mono duplication + norm)
    if output_as_stereo:
        if verbose:
            print("[stage] creating stereo output ...")
        out_stereo = torch.stack([out, out], dim=0)  # (2, n)
        ch_rms = torch.sqrt(torch.mean(out_stereo**2, dim=1) + 1e-12)
        scale = (target_rms / ch_rms).unsqueeze(1)
        out_stereo = out_stereo * scale.to(device=device, dtype=dtype)
        final_out = out_stereo
        final_shape = (2, n)
    else:
        final_out = out
        final_shape = (n,)

    if verbose:
        print("[done] enhancement complete.")

    return _to_output(final_out, audio, return_type), final_shape