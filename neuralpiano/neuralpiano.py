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
                output_audio_file,
                sample_rate=48000,
                denoising_steps=10,
                load_multi_instrumental_model=False,
                return_audio=False
               ):
    
    """
    Renders a MIDI file into mastered stereo audio using a neural synthesis and encoding-decoding pipeline.

    This function performs end-to-end audio rendering from a MIDI input, leveraging a SoundFont-based synthesizer
    followed by neural encoding, denoising, and mastering. It supports both single-instrument and multi-instrument
    models, and outputs a stereo waveform saved to disk. Optionally, the final audio can be returned as a NumPy array.

    Parameters:
    ----------
    input_midi_file : str
        Path to the input MIDI file (.mid) to be rendered.
    
    output_audio_file : str
        Path to the output audio file (.wav) where the final stereo waveform will be saved.
    
    sample_rate : int, optional (default=48000)
        Target sample rate for audio loading and output. Used during waveform decoding and mastering.
    
    denoising_steps : int, optional (default=10)
        Number of denoising refinement steps applied during neural decoding. Higher values may improve quality
        at the cost of compute time.
    
    load_multi_instrumental_model : bool, optional (default=False)
        If True, loads a model capable of handling multi-instrumental MIDI input. If False, assumes single-instrument
        (e.g., solo piano) rendering.
    
    return_audio : bool, optional (default=False)
        If True, returns the final stereo waveform as a NumPy array. Otherwise, only saves the audio to disk.

    Returns:
    -------
    stereo : np.ndarray, optional
        Stereo waveform of the rendered and mastered audio. Returned only if `return_audio=True`.

    Workflow:
    --------
    1. Loads a preconfigured SoundFont (.sf2) for initial MIDI synthesis.
    2. Synthesizes raw audio from MIDI using `midirenderer.render_wave_from`.
    3. Loads and resamples the waveform using `librosa`.
    4. Encodes the waveform into a latent representation via `EncoderDecoder.encode`.
    5. Decodes the latent signal with denoising using `EncoderDecoder.decode`.
    6. Applies mastering via `master_mono_piano` to produce stereo output.
    7. Saves the final audio using `soundfile.write`.

    Notes:
    -----
    - The SoundFont used is hardcoded as "SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2" and must be located in the
      "models" subdirectory of the current working directory.
    - Diagnostic print statements are included to trace each stage of the rendering pipeline.
    - Designed for modular integration and benchmarking in audio synthesis workflows.

    Example:
    --------
    >>> render_midi("input.mid", "output.wav", denoising_steps=15, return_audio=True)
    """
    
    home_root = os.getcwd()
    models_dir = os.path.join(home_root, "models")
    sf2_name = "SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"
    sf2_path = os.path.join(models_dir, sf2_name)

    print('=' * 70)
    print('Neural Piano')
    print('=' * 70)
    
    print('Prepping model...')
    encdec = EncoderDecoder(load_multi_instrumental_model=load_multi_instrumental_model)

    print('Reading and rendering MIDI file...')
    wav_data = midirenderer.render_wave_from(
        Path(sf2_path).read_bytes(),
        Path(input_midi_file).read_bytes()
    )
    
    print('Loading rendered MIDI...')
    with io.BytesIO(wav_data) as byte_stream:
        wv, sr = librosa.load(byte_stream, sr=sample_rate)
    
    print('Encoding...')
    latent = encdec.encode(wv)
    
    print('Rendering...')
    wv_rec = encdec.decode(latent, denoising_steps=denoising_steps)
    
    print('Mastering...')
    stereo, diag = master_mono_piano(wv_rec)
    
    print('Saving final audio...')
    sf.write(output_audio_file, stereo.squeeze().T, samplerate=sr)
    
    print('=' * 70)
    print('Done!')
    print('=' * 70)
    
    if return_audio:
        return stereo