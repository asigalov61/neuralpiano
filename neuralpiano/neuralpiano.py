# Neural Piano main Python module

import os
import io
from pathlib import Path
import importlib

import librosa
import soundfile as sf

import midirenderer

from .music2latent.inference import EncoderDecoder

from .master import master_mono_piano

def render_midi(input_midi_file,
                output_audio_file='neuralpiano_output.wav',
                sample_rate=48000,
                denoising_steps=10,
                use_v1_piano_model=False,
                load_multi_instrumental_model=False,
                custom_model_path=None,
                gain_db=10.0,
                sf2_name='SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2',
                trim_silence=True,
                trim_top_db=60,
                trim_frame_length=2048,
                trim_hop_length=512,
                return_audio=False,
                verbose=True
               ):
    """
    Render a MIDI file to a mastered stereo audio file using a neural piano encoder/decoder
    pipeline and a SoundFont-based MIDI renderer, with optional silence trimming and verbose logging.

    Summary
    - Reads a SoundFont file from a models directory relative to the current working directory.
    - Renders the provided MIDI file into raw waveform bytes using the MIDI renderer.
    - Loads the rendered waveform into memory using librosa at `sample_rate`.
    - Optionally trims leading/trailing silence with librosa.effects.trim.
    - Encodes the (optionally trimmed) waveform to a latent representation using EncoderDecoder.
    - Decodes the latent back to audio using optional denoising steps.
    - Applies mono-to-stereo mastering and gain to produce the final stereo waveform.
    - Writes the final stereo waveform to `output_audio_file` using soundfile (sf.write).
    - Optionally returns the final stereo waveform array.

    Parameters
    - input_midi_file (str or Path)
        Path to the input MIDI file to render. Must be readable.

    - output_audio_file (str or Path, default='neuralpiano_output.wav')
        Path where the final mastered audio will be saved. Format and bit depth are determined by
        the soundfile backend and file extension.

    - sample_rate (int, default=48000)
        Sampling rate (Hz) used for librosa.load and when writing the final audio.

    - denoising_steps (int, default=15)
        Number of refinement/denoising steps during decode. Higher values may yield cleaner audio
        at the expense of runtime.

    - use_v1_piano_model (bool, default=False)
        Toggle to initialize EncoderDecoder with the legacy v1 piano model variant.

    - load_multi_instrumental_model (bool, default=False)
        Toggle to initialize EncoderDecoder with a multi-instrument model variant.

    - custom_model_path (str or Path or None, default=None)
        Optional checkpoint path for model weights used at inference time.

    - gain_db (float, default=20.0)
        Gain (dB) applied during mastering.

    - sf2_name (str, default='SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2')
        SoundFont filename located under os.path.join(os.getcwd(), "models", sf2_name).

    - trim_silence (bool, default=True)
        If True, trims leading and trailing silence using librosa.effects.trim before encoding.

    - trim_top_db (float, default=60)
        dB threshold for silence trimming. Larger values trim more aggressively.

    - trim_frame_length (int, default=2048)
        Frame length (samples) used by librosa.effects.trim energy estimation.

    - trim_hop_length (int, default=512)
        Hop length (samples) used by librosa.effects.trim.

    - return_audio (bool, default=False)
        If True, the function returns the final mastered stereo waveform (numpy array).

    - verbose (bool, default=True)
        If True, prints progress and diagnostic messages for each pipeline stage; if False,
        suppresses non-error output to stdout. Important trimming diagnostics are only printed
        when verbose is True.

    Returns
    - None if return_audio is False.
    - numpy.ndarray (stereo waveform) if return_audio is True. The shape is pipeline-dependent
      (commonly (2, N) or (N, 2)) and the audio is sampled at `sample_rate`.

    Side effects
    - Reads the input MIDI and SF2 files from disk.
    - Writes a rendered audio file to `output_audio_file`.
    - Instantiates EncoderDecoder which may load model weights and consume substantial memory.
    - Uses librosa.load which converts raw bytes into a floating-point waveform array.

    Diagnostics and logging
    - When verbose is True, the function prints stage markers (prepping model, rendering,
      loading, trimming, encoding, decoding, mastering, and saving).
    - When trim_silence is True and verbose is True, prints trimmed sample interval and durations.

    Expected external symbols
    - EncoderDecoder(load_multi_instrumental_model, use_v1_piano_model, load_path_inference)
      with methods `.encode(waveform)` and `.decode(latent, denoising_steps=...)`.
    - midirenderer.render_wave_from(sf2_bytes, midi_bytes) -> raw WAV bytes.
    - master_mono_piano(mono_waveform, gain_db=...) -> (stereo_waveform, diagnostics).
    - Standard imports: os, io, Path (from pathlib), librosa, librosa.effects, soundfile as sf, numpy as np.

    Errors and exceptions
    - FileNotFoundError: if input MIDI or SF2 file is missing.
    - ValueError: if librosa, encoder/decoder, or mastering functions receive invalid inputs.
    - RuntimeError / MemoryError: possible with large models or limited resources.
    - soundfile write errors if output path is unwritable or disk is full.

    Implementation notes
    - SoundFont path resolved as os.path.join(os.getcwd(), "models", sf2_name). Use absolute paths
      or change working directory for reproducible behavior.
    - Trimming is conservative by default (trim_top_db=60) to keep natural release tails; adjust
      trim_top_db lower (e.g., 40-50) to preserve more quiet tails or raise it to trim more.
    - Verbose=False suppresses printed progress but does not suppress raised exceptions.

    Example
    >>> render_midi("arr.mid", "out.wav", sample_rate=44100, denoising_steps=20,
    ...             trim_silence=True, trim_top_db=55, verbose=True)
    >>> stereo = render_midi("arr.mid", "out.wav", return_audio=True, verbose=False)
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

    _pv('Reading and rendering MIDI file...')
    wav_data = midirenderer.render_wave_from(
        Path(sf2_path).read_bytes(),
        Path(input_midi_file).read_bytes()
    )

    _pv('Loading rendered MIDI...')
    with io.BytesIO(wav_data) as byte_stream:
        wv, sr = librosa.load(byte_stream, sr=sample_rate)

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

    _pv('Encoding...')
    latent = encdec.encode(wv)

    _pv('Rendering...')
    wv_rec = encdec.decode(latent, denoising_steps=denoising_steps)

    _pv('Mastering...')
    stereo, diag = master_mono_piano(wv_rec, gain_db=gain_db)

    _pv('Saving final audio...')
    sf.write(output_audio_file, stereo.squeeze().T, samplerate=sr)

    _pv('=' * 70)
    _pv('Done!')
    _pv('=' * 70)

    if return_audio:
        return stereo