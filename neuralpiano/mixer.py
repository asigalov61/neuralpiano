# Mixer Python module

import os
import math
from typing import List, Optional, Union, Sequence, Tuple
import numpy as np
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional progress bar
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

def _ensure_list_defaults(values, n, default=0.0):
    if values is None:
        return [default] * n
    if isinstance(values, (int, float)):
        return [float(values)] * n
    vals = list(values)
    if len(vals) >= n:
        return vals[:n]
    return vals + [default] * (n - len(vals))

def _resample_channel(channel, orig_sr, target_sr, res_type='kaiser_fast'):
    if orig_sr == target_sr:
        return channel
    return librosa.resample(channel, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type)

def _apply_fades(track: np.ndarray, sr: int, fade_in_s: float, fade_out_s: float) -> np.ndarray:
    n_samples = track.shape[1]
    if n_samples == 0:
        return track
    fi = max(0, int(round(fade_in_s * sr)))
    fo = max(0, int(round(fade_out_s * sr)))
    if fi + fo > n_samples and (fi + fo) > 0:
        scale = n_samples / (fi + fo)
        fi = int(round(fi * scale))
        fo = int(round(fo * scale))
    env = np.ones(n_samples, dtype=np.float32)
    if fi > 0:
        t = np.linspace(0.0, 1.0, fi, endpoint=True, dtype=np.float32)
        env[:fi] = 0.5 * (1.0 - np.cos(np.pi * t))
    if fo > 0:
        t = np.linspace(0.0, 1.0, fo, endpoint=True, dtype=np.float32)
        env[-fo:] = 0.5 * (1.0 + np.cos(np.pi * t))
    return track * env[np.newaxis, :]

def _normalize_input_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return np.expand_dims(arr.astype(np.float32), 0)
    if arr.ndim == 2:
        # Heuristic: if shape looks like (samples, channels) transpose to (channels, samples)
        if arr.shape[1] <= 2 and arr.shape[0] > arr.shape[1]:
            return arr.T.astype(np.float32)
        return arr.astype(np.float32)
    raise ValueError("Input numpy array must be 1D or 2D (samples,channels) or (channels,samples)")

def mix_audio(input_items: Sequence[Union[str, Tuple[np.ndarray, int]]],
              output_path: str = 'mixed.wav',
              gain_db: Optional[Union[float, List[float]]] = 0.0,
              pan: Optional[Union[float, List[float]]] = 0.0,
              delays: Optional[Union[float, List[float]]] = 0.0,
              fade_in: Optional[Union[float, List[float]]] = 0.0,
              fade_out: Optional[Union[float, List[float]]] = 0.0,
              target_sr: Optional[int] = None,
              normalize: bool = True,
              trim_trailing_silence: bool = False,
              trim_threshold_db: float = -60.0,
              trim_padding_seconds: float = 0.01,
              output_subtype: Optional[str] = None,
              workers: int = 4,
              show_progress: bool = False,
              verbose: bool = False,
              return_mix: bool = False
             ) -> Optional[Tuple[np.ndarray, int]]:
    
    """
    Mix inputs (file paths and/or (array, sr) tuples) into one audio file.
    If return_mix is True the function returns (mix_array, sample_rate) where mix_array
    is shaped (samples, channels) and dtype float32.

    All other parameters behave as in previous versions:
    - gain_db, pan, delays, fade_in, fade_out accept single values or shorter lists.
    - target_sr defaults to the highest input sample rate.
    - normalize scales final peak to 0.999 when True.
    - trim_trailing_silence removes trailing silence below trim_threshold_db.
    - workers controls parallel processing.
    - show_progress uses tqdm if available.
    - verbose prints progress messages.
    """
    
    n = len(input_items)
    if n == 0:
        raise ValueError("input_items must contain at least one element")

    # Prepare per-track parameter lists (allow shorter lists)
    gains_db = _ensure_list_defaults(gain_db, n, default=0.0)
    pans = _ensure_list_defaults(pan, n, default=0.0)
    delays_sec = _ensure_list_defaults(delays, n, default=0.0)
    fades_in = _ensure_list_defaults(fade_in, n, default=0.0)
    fades_out = _ensure_list_defaults(fade_out, n, default=0.0)
    pans = [max(-1.0, min(1.0, float(p))) for p in pans]

    if verbose:
        print(f"[mix_audio] Preparing {n} inputs...")

    tracks = []
    srs = []
    channel_counts = []

    load_iter = list(enumerate(input_items))
    if show_progress and _HAS_TQDM:
        load_iter = list(tqdm(load_iter, desc="Loading inputs", unit="item"))

    for idx, item in load_iter:
        if isinstance(item, str):
            if verbose:
                print(f"  loading file [{idx+1}/{n}]: {item}")
            y, sr = librosa.load(item, sr=None, mono=False)
            if y.ndim == 1:
                y = np.expand_dims(y, 0)
            tracks.append(y.astype(np.float32))
            srs.append(sr)
            channel_counts.append(y.shape[0])
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            arr, sr = item
            if verbose:
                print(f"  using array input [{idx+1}/{n}] with sr={sr}")
            y = _normalize_input_array(arr)
            tracks.append(y.astype(np.float32))
            srs.append(int(sr))
            channel_counts.append(y.shape[0])
        else:
            raise ValueError("Each input item must be a file path or a tuple (array, sr)")

    # Decide target sample rate
    if target_sr is None:
        target_sr = max(srs)
    if verbose:
        print(f"[mix_audio] Target sample rate: {target_sr} Hz")

    # Determine output channel count: stereo if any input has >=2 channels, else mono
    out_ch = 2 if any(c >= 2 for c in channel_counts) else 1
    if verbose:
        print(f"[mix_audio] Output channels: {out_ch}")

    # Process each track: resample, channel handling, panning, fades, apply gain
    processed = [None] * n

    def process_track(i):
        t = tracks[i]
        sr = srs[i]
        g_db = gains_db[i]
        p_val = pans[i]
        delay = float(delays_sec[i])
        fi = float(fades_in[i])
        fo = float(fades_out[i])

        # Downmix multichannel (>2) to mono first
        if t.shape[0] > 2:
            t = np.expand_dims(np.mean(t, axis=0), 0)

        # Resample channels
        if sr != target_sr:
            if workers > 1 and t.shape[0] > 1:
                with ThreadPoolExecutor(max_workers=min(workers, t.shape[0])) as ex:
                    futures = [ex.submit(_resample_channel, t[ch], sr, target_sr, 'kaiser_fast') for ch in range(t.shape[0])]
                    resampled_ch = [f.result() for f in futures]
                t = np.vstack(resampled_ch)
            else:
                t = np.vstack([_resample_channel(t[ch], sr, target_sr, 'kaiser_fast') for ch in range(t.shape[0])])

        # Channel handling and panning
        if t.shape[0] == 1 and out_ch == 2:
            mono = t[0]
            angle = (p_val + 1.0) * (math.pi / 4.0)
            left_gain = math.cos(angle)
            right_gain = math.sin(angle)
            left = mono * left_gain
            right = mono * right_gain
            t = np.vstack([left, right])
        elif t.shape[0] == 2 and out_ch == 2:
            left, right = t[0], t[1]
            mono = 0.5 * (left + right)
            angle = (p_val + 1.0) * (math.pi / 4.0)
            left_gain = math.cos(angle)
            right_gain = math.sin(angle)
            t = np.vstack([mono * left_gain, mono * right_gain])
        elif t.shape[0] == 2 and out_ch == 1:
            mono = 0.5 * (t[0] + t[1])
            t = np.expand_dims(mono, 0)

        # Apply fades (in seconds)
        if (fi > 0.0) or (fo > 0.0):
            t = _apply_fades(t, target_sr, fi, fo)

        # Apply gain (dB -> linear)
        lin = 10.0 ** (g_db / 20.0)
        t = t * lin

        # Convert delay seconds to sample offset (can be negative)
        offset_samples = int(round(delay * target_sr))
        return t, offset_samples

    if verbose:
        print("[mix_audio] Resampling and processing tracks...")

    # Parallel processing of tracks
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_track, i): i for i in range(n)}
            if show_progress and _HAS_TQDM:
                pbar = tqdm(total=n, desc="Processing", unit="track")
            for fut in as_completed(futures):
                i = futures[fut]
                processed[i] = fut.result()
                if show_progress and _HAS_TQDM:
                    pbar.update(1)
            if show_progress and _HAS_TQDM:
                pbar.close()
    else:
        proc_iter = range(n)
        if show_progress and _HAS_TQDM:
            proc_iter = tqdm(proc_iter, desc="Processing", unit="track")
        for i in proc_iter:
            processed[i] = process_track(i)

    # Determine final mix length considering offsets
    end_positions = []
    for t, offset in processed:
        start = max(0, offset)
        end = start + t.shape[1]
        end_positions.append(end)
    max_len = max(end_positions) if end_positions else 0
    min_start = min(offset for _, offset in processed)

    # If there are negative offsets, shift everything forward so earliest sample >= 0
    shift_forward = 0
    if min_start < 0:
        shift_forward = -min_start
        max_len += shift_forward
        if verbose:
            print(f"[mix_audio] Negative offsets detected. Shifting all tracks forward by {shift_forward} samples")

    # Create mix buffer and add tracks at offsets
    mix = np.zeros((out_ch, max_len), dtype=np.float32)
    if show_progress and _HAS_TQDM:
        mix_iter = tqdm(processed, desc="Mixing", unit="track")
    else:
        mix_iter = processed

    for (t, offset) in mix_iter:
        start = offset + shift_forward
        if start < 0:
            clip = -start
            if clip >= t.shape[1]:
                continue
            t = t[:, clip:]
            start = 0
        end = start + t.shape[1]
        mix[:, start:end] += t

    # Normalize to avoid clipping
    if normalize:
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix *= (0.999 / peak)
            if verbose:
                print(f"[mix_audio] Normalized by factor {(0.999/peak):.6f}")

    # Optional trailing silence removal
    if trim_trailing_silence:
        if verbose:
            print(f"[mix_audio] Trimming trailing silence below {trim_threshold_db} dBFS")
        threshold_lin = 10.0 ** (trim_threshold_db / 20.0)
        abs_max = np.max(np.abs(mix), axis=0)
        non_silent = np.where(abs_max > threshold_lin)[0]
        if non_silent.size > 0:
            last_idx = int(non_silent[-1])
            pad_samples = int(round(trim_padding_seconds * target_sr))
            new_len = min(mix.shape[1], last_idx + 1 + pad_samples)
            mix = mix[:, :new_len]
            if verbose:
                print(f"[mix_audio] Trimmed to {new_len} samples ({new_len/target_sr:.3f} s)")
        else:
            keep = int(round(trim_padding_seconds * target_sr))
            mix = mix[:, :keep]
            if verbose:
                print(f"[mix_audio] All silent; keeping {keep} samples ({keep/target_sr:.3f} s)")

    out = mix.T  # (samples, channels)

    # Infer format from extension and validate
    ext = os.path.splitext(output_path)[1].lower().lstrip('.')
    fmt = ext.upper()
    available = sf.available_formats()
    if fmt not in available:
        raise ValueError(f"Output format {fmt} not supported by soundfile. Available: {list(available.keys())}")

    if return_mix:
        print(f"[mix_audio] Returning output")
        # Return a copy to avoid accidental modification of internal buffer
        return out.copy(), int(target_sr)

    if verbose:
        print(f"[mix_audio] Writing output to {output_path} (format={fmt}, subtype={output_subtype})")

    sf.write(output_path, out, samplerate=target_sr, format=fmt, subtype=output_subtype)

    if verbose:
        print("[mix_audio] Done.")

    return output_path