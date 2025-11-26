#===============================================================================
# Neural Piano main Python module
#===============================================================================

"""

This module exposes `render_midi`, a high-level convenience function that:
- Renders a MIDI file to a raw waveform using a SoundFont (SF2) via `midirenderer`.
- Loads the rendered waveform and optionally trims silence.
- Encodes the waveform into a latent representation and decodes it using a
  learned Encoder/Decoder model to produce a high-quality piano audio render.
- Optionally applies a sequence of post-processing steps: denoising,
  bass enhancement, full-spectrum enhancement, and mastering.
- Writes the final audio to disk or returns it in-memory.

Design goals
------------
- Provide a single, well-documented function to convert MIDI -> polished WAV.
- Keep sensible defaults so the function works out-of-the-box with a
  `models/` directory containing the required SoundFont and model artifacts.
- Allow advanced users to override model paths, processing parameters, and
  device selection.

Dependencies
------------
- Python 3.8+
- torch
- librosa
- soundfile (pysoundfile)
- midirenderer
- The package's internal modules:
    - .music2latent.inference.EncoderDecoder
    - .denoise.denoise_audio
    - .bass.enhance_audio_bass
    - .enhancer.enhance_audio_full
    - .master.master_mono_piano

Typical usage
-------------
>>> from neuralpiano.main import render_midi
>>> out_path = render_midi("score.mid")  # writes ./score.wav using defaults
>>> audio, sr = render_midi("score.mid", return_audio=True)  # get numpy array and sr

Notes and behavior
------------------
- By default the function expects a `models/` directory in the current working
  directory and a SoundFont file named `SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2`.
  Use `sf2_name` or `custom_model_path` to override.
- The EncoderDecoder is instantiated per-call. For repeated renders in a
  long-running process, consider reusing a single EncoderDecoder instance
  (not provided by this convenience wrapper).
- `sample_rate` controls the resampling rate used when loading the rendered
  MIDI waveform and is propagated to downstream processing where relevant.
- `trim_silence` uses `librosa.effects.trim` with configurable `trim_top_db`,
  `trim_frame_length`, and `trim_hop_length`.
- Post-processing steps are applied in this order when enabled:
    1. denoise (if `denoise=True`)
    2. bass enhancement (if `enhance_bass=True`)
    3. full enhancement (if `enhance_full=True`)
    4. mastering (if `master=True`)
  Each step accepts a kwargs dict (e.g., `denoise_kwargs`) to override defaults.
- `device` accepts any value accepted by `torch.device` (e.g., 'cuda', 'cpu').
- When `return_audio=True`, the function returns `(final_audio, sample_rate)` as
  a NumPy array and sample rate. Otherwise it writes a WAV file and returns the
  output file path.
- Verbosity:
    - `verbose` prints high-level progress messages.
    - `verbose_diag` prints additional diagnostic values useful for debugging.

Exceptions
----------
The function may raise exceptions originating from:
- File I/O (missing MIDI or models, permission errors).
- `midirenderer` if the SoundFont or MIDI bytes are invalid.
- `librosa` when loading or trimming audio.
- Torch/model-related errors when instantiating or running the EncoderDecoder.
- Any of the post-processing modules if they encounter invalid inputs.

Parameters
----------
input_midi_file : str
    Path to the input MIDI file to render.
output_audio_file : str or None
    Path to write the final WAV file. If None, the output filename is derived
    from the MIDI filename and written to the current working directory.
sample_rate : int
    Target sample rate for loading and processing audio (default: 48000).
denoising_steps : int
    Default number of denoising steps passed to the decoder.
max_batch_size : int or None
    Maximum batch size to use when encoding/decoding; passed to EncoderDecoder.
use_v1_piano_model : bool
    If True, instructs EncoderDecoder to use the v1 piano model variant.
load_multi_instrumental_model : bool
    If True, load a multi-instrument model variant (if available).
custom_model_path : str or None
    Path to a custom model checkpoint for inference; passed to EncoderDecoder.
sf2_name : str
    Filename of the SoundFont inside the `models/` directory (default provided).
trim_silence : bool
    If True, trim leading/trailing silence from the rendered waveform.
trim_top_db : float
    `top_db` parameter for `librosa.effects.trim` (default: 60).
trim_frame_length : int
    `frame_length` parameter for `librosa.effects.trim` (default: 2048).
trim_hop_length : int
    `hop_length` parameter for `librosa.effects.trim` (default: 512).
denoise : bool
    If True, run the denoiser post-processing step.
denoise_kwargs : dict or None
    Additional keyword arguments for `denoise_audio`.
enhance_bass : bool
    If True, run bass enhancement post-processing.
bass_kwargs : dict or None
    Additional keyword arguments for `enhance_audio_bass`. `low_gain_db` is set
    from the top-level `low_gain_db` unless overridden here.
low_gain_db : float
    Default low-frequency gain (dB) used by bass enhancement (default: 8.0).
enhance_full : bool
    If True, run full-spectrum enhancement post-processing.
enhance_full_kwargs : dict or None
    Additional keyword arguments for `enhance_audio_full`.
master : bool
    If True, run final mastering pass.
master_kwargs : dict or None
    Additional keyword arguments for `master_mono_piano`. `gain_db` is set from
    the top-level `overall_gain_db` unless overridden here.
overall_gain_db : float
    Default gain (dB) applied during mastering (default: 10.0).
device : str
    Device string for torch (e.g., 'cuda' or 'cpu'). Converted to `torch.device`
    when passed to post-processing functions.
return_audio : bool
    If True, return `(audio_numpy, sample_rate)` instead of writing a file.
verbose : bool
    Print progress messages when True.
verbose_diag : bool
    Print diagnostic information when True.

Example
-------
Render a MIDI to disk with default processing:

>>> render_midi("song.mid")

Render and receive audio in-memory without post-processing:

>>> audio, sr = render_midi("song.mid", denoise=False, enhance_bass=False,
...                         enhance_full=False, master=False, return_audio=True)

"""

#===============================================================================

import os
import io
from pathlib import Path

import torch

import librosa
import soundfile as sf

import midirenderer

from .music2latent.inference import EncoderDecoder

from .denoise import denoise_audio

from .bass import enhance_audio_bass

from .enhancer import enhance_audio_full

from .master import master_mono_piano

#===============================================================================

def render_midi(input_midi_file,
                output_audio_file=None,
                sample_rate=48000,
                denoising_steps=10,
                max_batch_size=None,
                use_v1_piano_model=False,
                load_multi_instrumental_model=False,
                custom_model_path=None,
                sf2_name='SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2',
                trim_silence=True,
                trim_top_db=60,
                trim_frame_length=2048,
                trim_hop_length=512,
                denoise=True,
                denoise_kwargs=None,
                enhance_bass=False,
                bass_kwargs=None,
                low_gain_db=8.0,
                enhance_full=True,
                enhance_full_kwargs=None,
                master=False,
                master_kwargs=None,
                overall_gain_db=10.0,
                device='cuda',
                return_audio=False,
                verbose=True,
                verbose_diag=False
               ):
    
    """
    Render a MIDI file to a polished piano audio waveform or WAV file.

    This function orchestrates the full Neural Piano pipeline:
      1. Render MIDI -> raw waveform using a SoundFont via `midirenderer`.
      2. Load and optionally trim silence from the rendered waveform.
      3. Encode waveform to latent and decode with the model to synthesize audio.
      4. Optionally apply denoising, bass enhancement, full enhancement, and mastering.
      5. Return the final audio as a NumPy array or write it to disk as a WAV file.

    Parameters
    ----------
    input_midi_file : str
        Path to the input MIDI file.
    output_audio_file : str or None
        Path to write the final WAV file. If None, derived from MIDI filename.
    sample_rate : int
        Sample rate for loading and processing audio (default: 48000).
    denoising_steps : int
        Default denoising steps passed to decoder.
    max_batch_size : int or None
        Max batch size for EncoderDecoder operations.
    use_v1_piano_model : bool
        Use v1 piano model variant if True.
    load_multi_instrumental_model : bool
        Load multi-instrument model variant if True.
    custom_model_path : str or None
        Path to a custom model checkpoint for inference.
    sf2_name : str
        SoundFont filename located in the `models/` directory.
    trim_silence : bool
        Trim leading/trailing silence from the rendered waveform.
    trim_top_db : float
        `top_db` for `librosa.effects.trim`.
    trim_frame_length : int
        `frame_length` for `librosa.effects.trim`.
    trim_hop_length : int
        `hop_length` for `librosa.effects.trim`.
    denoise : bool
        Run denoiser post-processing when True.
    denoise_kwargs : dict or None
        Extra kwargs for `denoise_audio`.
    enhance_bass : bool
        Run bass enhancement when True.
    bass_kwargs : dict or None
        Extra kwargs for `enhance_audio_bass`. `low_gain_db` is set from the
        top-level argument unless overridden here.
    low_gain_db : float
        Default low-frequency gain (dB) for bass enhancement.
    enhance_full : bool
        Run full-spectrum enhancement when True.
    enhance_full_kwargs : dict or None
        Extra kwargs for `enhance_audio_full`.
    master : bool
        Run final mastering when True.
    master_kwargs : dict or None
        Extra kwargs for `master_mono_piano`. `gain_db` is set from the top-level
        argument unless overridden here.
    overall_gain_db : float
        Default gain (dB) applied during mastering.
    device : str
        Torch device string (e.g., 'cuda' or 'cpu').
    return_audio : bool
        If True, return `(audio_numpy, sample_rate)` instead of writing a file.
    verbose : bool
        Print progress messages.
    verbose_diag : bool
        Print diagnostic information for debugging.

    Returns
    -------
    str or (numpy.ndarray, int)
        If `return_audio` is False (default), returns the path to the written WAV file.
        If `return_audio` is True, returns a tuple `(audio_numpy, sample_rate)` where
        `audio_numpy` is a 1-D NumPy array (mono) and `sample_rate` is an int.

    Raises
    ------
    FileNotFoundError
        If the input MIDI file or required model/SF2 files are missing.
    RuntimeError
        If model inference or post-processing fails (propagates underlying errors).

    """
    
    def _pv(msg):
        if verbose:
            print(msg)

    _pv('=' * 70)
    _pv('Neural Piano')
    _pv('=' * 70)

    # Normalize kwargs buckets
    denoise_kwargs = {} if denoise_kwargs is None else dict(denoise_kwargs)
    bass_kwargs = {} if bass_kwargs is None else dict(bass_kwargs)
    enhance_full_kwargs = {} if enhance_full_kwargs is None else dict(enhance_full_kwargs)
    master_kwargs = {} if master_kwargs is None else dict(master_kwargs)

    # Provide sensible defaults from top-level args unless overridden in kwargs
    if 'low_gain_db' not in bass_kwargs:
        bass_kwargs['low_gain_db'] = low_gain_db

    if 'overall_gain_db' not in enhance_full_kwargs:
        enhance_full_kwargs['overall_gain_db'] = overall_gain_db

    if 'gain_db' not in master_kwargs:
        master_kwargs['gain_db'] = overall_gain_db

    home_root = os.getcwd()
    models_dir = os.path.join(home_root, "models")
    sf2_path = os.path.join(models_dir, sf2_name)

    if verbose_diag:
        _pv(home_root)
        _pv(models_dir)
        _pv(sf2_path)
        _pv('=' * 70)

    _pv('Prepping model...')
    encdec = EncoderDecoder(load_multi_instrumental_model=load_multi_instrumental_model,
                            use_v1_piano_model=use_v1_piano_model,
                            load_path_inference=custom_model_path
                            )

    if verbose_diag:
        try:
            _pv(encdec.gen)
        except Exception:
            _pv('encdec.gen: <unavailable>')
        _pv('=' * 70)

    _pv('Reading and rendering MIDI file...')
    wav_data = midirenderer.render_wave_from(
        Path(sf2_path).read_bytes(),
        Path(input_midi_file).read_bytes()
    )

    if verbose_diag:
        _pv(len(wav_data))
        _pv('=' * 70)

    _pv('Loading rendered MIDI...')
    with io.BytesIO(wav_data) as byte_stream:
        wv, sr = librosa.load(byte_stream, sr=sample_rate)

    if verbose_diag:
        _pv(sr)
        _pv(wv.shape)
        _pv('=' * 70)

    if trim_silence:
        _pv('Trimming leading and trailing silence from rendered waveform...')
        wv_trimmed, trim_interval = librosa.effects.trim(
            wv,
            top_db=trim_top_db,
            frame_length=trim_frame_length,
            hop_length=trim_hop_length
        )
        start_sample, end_sample = trim_interval
        orig_dur = len(wv) / sr
        trimmed_dur = len(wv_trimmed) / sr
        if verbose:
            _pv(f'  Trimmed samples: start={start_sample}, end={end_sample}')
            _pv(f'  Duration before={orig_dur:.3f}s, after={trimmed_dur:.3f}s')
        wv = wv_trimmed
    else:
        _pv('Silence trimming disabled; using full rendered waveform.')

    if verbose_diag:
        _pv(wv.shape)
        _pv('=' * 70)

    _pv('Encoding...')
    latent = encdec.encode(wv,
                           max_batch_size=max_batch_size,
                           show_progress=verbose
                           )

    if verbose_diag:
        try:
            _pv(latent.shape)
        except Exception:
            _pv('latent.shape: <unavailable>')
        _pv('=' * 70)

    _pv('Rendering...')
    audio = encdec.decode(latent,
                          denoising_steps=denoising_steps,
                          max_batch_size=max_batch_size,
                          show_progress=verbose
                         )

    audio = audio.squeeze()

    if verbose_diag:
        try:
            _pv(audio.shape)
        except Exception:
            _pv('audio.shape: <unavailable>')
        _pv('=' * 70)

    # Post-processing: denoise
    if denoise:
        _pv('Denoising...')
        # Always pass sr and device; allow denoise_kwargs to override them if provided
        denoise_call_kwargs = dict(sr=sr, device=torch.device(device))
        denoise_call_kwargs.update(denoise_kwargs)
        audio, den_diag = denoise_audio(audio, **denoise_call_kwargs)

        if verbose_diag:
            _pv(den_diag)
            _pv('=' * 70)

    # Post-processing: bass enhancement
    if enhance_bass:
        _pv('Enhancing bass...')
        bass_call_kwargs = dict(sr=sr, device=torch.device(device))
        bass_call_kwargs.update(bass_kwargs)
        audio, bass_diag = enhance_audio_bass(audio, **bass_call_kwargs)

        if verbose_diag:
            _pv(bass_diag)
            _pv('=' * 70)

    # Post-processing: full enhancement (placed before mastering)
    if enhance_full:
        _pv('Enhancing full audio...')
        full_call_kwargs = dict(sr=sr, device=torch.device(device))
        full_call_kwargs.update(enhance_full_kwargs)
        
        if not master:
            output_as_stereo = True
            
        else:
            output_as_stereo = False
        
        audio, full_diag = enhance_audio_full(audio,
                                              output_as_stereo=output_as_stereo,
                                              **full_call_kwargs
                                              )

        if verbose_diag:
            _pv(full_diag)
            _pv('=' * 70)

    # Post-processing: mastering
    if master:
        _pv('Mastering...')
        master_call_kwargs = dict(device=torch.device(device))
        master_call_kwargs.update(master_kwargs)
        audio, mas_diag = master_mono_piano(audio, **master_call_kwargs)

        if verbose_diag:
            _pv(mas_diag)
            _pv('=' * 70)

    if verbose_diag:
        try:
            _pv(audio.shape)
        except Exception:
            _pv('audio.shape: <unavailable>')
        _pv('=' * 70)

    _pv('Creating final audio...')
    final_audio = audio.cpu().numpy().squeeze().T

    if verbose_diag:
        _pv(final_audio.shape)
        _pv(sr)
        _pv('=' * 70)

    if return_audio:
        _pv('Returning final audio...')
        _pv('=' * 70)
        _pv('Done!')
        _pv('=' * 70)
        return final_audio, sr

    else:
        _pv('Saving final audio...')
        if output_audio_file is None:
            midi_name = os.path.basename(input_midi_file)
            output_name, _ = os.path.splitext(midi_name)
            output_audio_file = os.path.join(home_root, output_name + '.wav')

        if verbose_diag:
            _pv(output_audio_file)
            _pv(sr)
            _pv('=' * 70)

        sf.write(output_audio_file, final_audio, samplerate=sr)

        _pv('=' * 70)
        _pv('Done!')
        _pv('=' * 70)

        return output_audio_file