# Neural Piano main Python module

"""
Neural Piano - MIDI to high-quality piano audio renderer.

This module provides a single high-level function `render_midi` that converts a
MIDI file into a polished, mastered piano audio file using a sample-based MIDI
renderer followed by a learned music-to-latent encoder/decoder and optional
post-processing (denoising, bass enhancement, and mastering).

Design goals
------------
- Provide a simple, script-friendly entry point for converting MIDI -> WAV.
- Use a SoundFont (SF2) to render MIDI to a raw waveform, then apply a neural
  encoder/decoder to synthesize a high-fidelity piano timbre.
- Offer configurable preprocessing and postprocessing steps so users can tune
  quality vs. speed and integrate into batch workflows.

High-level pipeline
-------------------
1. Locate the SoundFont file (SF2) in a `models/` directory under the current
   working directory and read the input MIDI file bytes.
2. Use `midirenderer.render_wave_from` to render the MIDI into a raw waveform.
3. Optionally trim leading/trailing silence using `librosa.effects.trim`.
4. Encode the rendered waveform into a latent representation using the
   `EncoderDecoder`'s `encode` method.
5. Decode the latent representation back to audio with `EncoderDecoder.decode`.
6. Optionally apply:
   - Denoising (`denoise_audio`)
   - Bass enhancement (`enhance_bass`)
   - Mastering (`master_mono_piano`)
7. Save the final audio to disk (default: WAV) and optionally return the final
   audio numpy array.

Notes
-----
- The function expects a `models/` directory in the current working directory
  containing the requested SF2 file (default: 'SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2').
- The neural encoder/decoder is instantiated from `.music2latent.inference.EncoderDecoder`.
  You can pass `custom_model_path` to load a specific trained model for inference.
- Post-processing functions (`denoise_audio`, `enhance_bass`, `master_mono_piano`)
  are expected to accept and return PyTorch tensors on the specified device.
- The function uses `librosa` for audio loading and trimming and `soundfile` for
  writing the final WAV file.

Limitations and assumptions
---------------------------
- The function assumes a CUDA-capable device by default (`device='cuda'`). If
  CUDA is not available, pass `device='cpu'`.
- The function reads the entire MIDI and SF2 into memory; extremely large SF2
  or MIDI files may increase memory usage.
- The function writes the output file using `soundfile.write` and will overwrite
  an existing file at `output_audio_file` without prompting.

Example
-------
>>> render_midi("example.mid", output_audio_file="out.wav", sample_rate=48000,
...             denoising_steps=12, denoise=True, master=True, device="cpu")
"""

import os
import io
from pathlib import Path
import importlib

import torch

import librosa
import soundfile as sf

import midirenderer

from .music2latent.inference import EncoderDecoder

from .denoise import denoise_audio

from .bass import enhance_audio_bass

from .master import master_mono_piano


def render_midi(input_midi_file,
                output_audio_file='neuralpiano_output.wav',
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
                denoise=False,
                enhance_bass=True,
                low_gain_db=8.0,
                master=True,
                gain_db=10.0,
                device='cuda',
                return_audio=False,
                verbose=True,
                verbose_diag=False
               ):
    
    """
    Render a MIDI file to a polished piano audio file.

    This function orchestrates a multi-stage pipeline:
      1. Render MIDI to a raw waveform using a SoundFont (SF2) via midirenderer.
      2. Optionally trim silence from the rendered waveform.
      3. Encode the waveform into a latent representation using a neural encoder.
      4. Decode the latent representation into a high-quality audio waveform.
      5. Optionally apply denoising, bass enhancement, and mastering.
      6. Save the final audio to disk and optionally return it as a NumPy array.

    Parameters
    ----------
    input_midi_file : str or pathlib.Path
        Path to the input MIDI file to render.
    output_audio_file : str, optional
        Path where the final audio file will be written (default 'neuralpiano_output.wav').
    sample_rate : int, optional
        Target sample rate for audio processing and output (default 48000).
    denoising_steps : int, optional
        Number of denoising steps passed to the decoder (if applicable) (default 10).
    max_batch_size : int or None, optional
        Maximum batch size to use for encoding/decoding. If None, the encoder/decoder
        will choose a default or compute an appropriate batch size (default None).
    use_v1_piano_model : bool, optional
        If True, instruct the EncoderDecoder to use a legacy v1 piano model variant
        (default False).
    load_multi_instrumental_model : bool, optional
        If True, load a multi-instrument model variant in the EncoderDecoder (default False).
    custom_model_path : str or None, optional
        Path to a custom model checkpoint to load for inference. If None, the default
        packaged model is used (default None).
    sf2_name : str, optional
        Filename of the SoundFont (SF2) to use for MIDI rendering. The file is
        expected to be located in a `models/` directory under the current working
        directory (default 'SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2').
    trim_silence : bool, optional
        If True, trim leading and trailing silence from the rendered waveform using
        `librosa.effects.trim` (default True).
    trim_top_db : float, optional
        The `top_db` parameter passed to `librosa.effects.trim` controlling the
        silence threshold in decibels (default 60).
    trim_frame_length : int, optional
        Frame length parameter for `librosa.effects.trim` (default 2048).
    trim_hop_length : int, optional
        Hop length parameter for `librosa.effects.trim` (default 512).
    denoise : bool, optional
        If True, run the denoising post-processing step (default True).
    enhance_bass : bool, optional
        If True, run the bass enhancement post-processing step (default True).
    low_gain_db : float, optional
        Gain in dB applied to low frequencies during bass enhancement (default 8.0).
    master : bool, optional
        If True, run the mastering post-processing step (default True).
    gain_db : float, optional
        Mastering gain in dB applied during the mastering step (default 10.0).
    device : str or torch.device, optional
        Device identifier for PyTorch operations (e.g., 'cuda' or 'cpu') (default 'cuda').
    return_audio : bool, optional
        If True, return the final audio as a NumPy array in addition to writing the file
        (default False).
    verbose : bool, optional
        If True, print high-level progress messages to stdout (default True).
    verbose_diag : bool, optional
        If True, print diagnostic information (shapes, intermediate diagnostics) to stdout
        for debugging (default False).

    Returns
    -------
    numpy.ndarray or None
        If `return_audio` is True, returns the final audio as a NumPy array with shape
        (n_samples,) or (n_channels, n_samples) depending on processing. If `return_audio`
        is False, returns None.

    Side effects
    ------------
    - Reads the SF2 file from `./models/{sf2_name}`.
    - Reads the MIDI file from `input_midi_file`.
    - Writes the final audio to `output_audio_file` (overwrites if exists).
    - Instantiates and loads neural model weights via `EncoderDecoder`.

    Errors and exceptions
    ---------------------
    - FileNotFoundError: if the SF2 file or MIDI file cannot be found/read.
    - RuntimeError: may be raised by PyTorch if the requested `device` is invalid or
      if model loading fails.
    - Any exceptions raised by `midirenderer`, `librosa`, or the post-processing
      functions will propagate to the caller.

    Implementation details
    ----------------------
    - The function constructs the SF2 path by joining the current working directory
      with a `models` subdirectory. This keeps model assets colocated with the
      working project.
    - The raw MIDI rendering is performed in-memory: `midirenderer.render_wave_from`
      returns raw WAV bytes which are loaded into `librosa` via an `io.BytesIO`.
    - The neural pipeline expects and returns PyTorch tensors; before writing the
      final file the tensor is moved to CPU and converted to a NumPy array.
    - Diagnostic prints are guarded by `verbose` and `verbose_diag` flags.

    Example usage
    -------------
    Basic render and save:

    >>> render_midi("song.mid", "song_out.wav")

    Render with custom model, no trimming, and return audio array:

    >>> audio = render_midi("song.mid",
    ...                     output_audio_file="song_out.wav",
    ...                     custom_model_path="checkpoints/my_model.pt",
    ...                     trim_silence=False,
    ...                     return_audio=True,
    ...                     device="cpu")

    """
    
    home_root = os.getcwd()
    models_dir = os.path.join(home_root, "models")
    sf2_path = os.path.join(models_dir, sf2_name)

    def _pv(msg):
        if verbose:
            print(msg)

    _pv('=' * 70)
    _pv('Neural Piano')
    _pv('=' * 70)

    _pv('Prepping model...')
    encdec = EncoderDecoder(load_multi_instrumental_model=load_multi_instrumental_model,
                            use_v1_piano_model=use_v1_piano_model,
                            load_path_inference=custom_model_path
                            )

    if verbose_diag:
        _pv(encdec.gen)
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
        _pv(latent.shape)
        _pv('=' * 70)

    _pv('Rendering...')
    audio = encdec.decode(latent,
                          denoising_steps=denoising_steps,
                          max_batch_size=max_batch_size,
                          show_progress=verbose
                         )
    
    audio = audio.squeeze()

    if verbose_diag:
        _pv(audio.shape)
        _pv('=' * 70)

    if denoise:
        _pv('Denoising...')
        audio, den_diag = denoise_audio(audio,
                                        sr=sr,
                                        device=torch.device(device)
                                        )

        if verbose_diag:
            _pv(den_diag)
            _pv('=' * 70)

    if enhance_bass:
        _pv('Enhancing bass...')
        audio, bass_diag = enhance_audio_bass(audio,
                                              sr=sr,
                                              low_gain_db=low_gain_db,
                                              device=torch.device(device)
                                             )

        if verbose_diag:
            _pv(bass_diag)
            _pv('=' * 70)

    if master:
        _pv('Mastering...')
        audio, mas_diag = master_mono_piano(audio,
                                            gain_db=gain_db,
                                            device=torch.device(device)
                                           )

        if verbose_diag:
            _pv(mas_diag)
            _pv('=' * 70)

    if verbose_diag:
        _pv(audio.shape)
        _pv('=' * 70)

    _pv('Saving final audio...')
    final_audio = audio.cpu().numpy().squeeze().T

    if verbose_diag:
        _pv(final_audio.shape)
        _pv('=' * 70)

    sf.write(output_audio_file, final_audio, samplerate=sr)

    _pv('=' * 70)
    _pv('Done!')
    _pv('=' * 70)

    if return_audio:
        return final_audio