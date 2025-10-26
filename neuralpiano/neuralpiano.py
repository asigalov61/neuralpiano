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
                denoising_steps=15,
                load_multi_instrumental_model=False,
                gain_db=20.0,
                sf2_name='SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2',
                return_audio=False
               ):
    
    """
    Render a MIDI file into a mastered stereo audio file using a neural piano encoder-decoder pipeline and a SoundFont.

    This function performs a full, end-to-end render of a MIDI file by:
    1. Locating and reading a SoundFont (SF2) from a local models directory.
    2. Rendering the MIDI into a raw WAV via a MIDI renderer that consumes the SF2 bytes and MIDI bytes.
    3. Loading the rendered WAV into memory at the requested sample rate.
    4. Encoding the mono waveform with a neural EncoderDecoder.
    5. Decoding the latent representation with configurable denoising to produce a reconstructed waveform.
    6. Applying a mastering chain for mono piano to produce a stereo, gain-matched output and diagnostics.
    7. Writing the final stereo audio to disk and optionally returning the stereo audio array.

    Parameters
    - input_midi_file (str or pathlib.Path)
      Path to the input MIDI file to be rendered.

    - output_audio_file (str or pathlib.Path)
      Path where the final mastered stereo audio will be saved. The function writes with the samplerate used to load the rendered MIDI.

    - sample_rate (int, default 48000)
      Target sample rate used when loading the intermediate rendered WAV into memory. The encoder/decoder and master expect the audio at this sample rate.

    - denoising_steps (int, default 15)
      Number of denoising steps passed to the decoder when reconstructing audio from latent space. Higher values generally produce smoother output at the cost of more processing.

    - load_multi_instrumental_model (bool, default False)
      Whether to initialize the EncoderDecoder in a multi-instrumental mode. Use True when rendering MIDI that contains multiple instrument tracks and the model supports it.

    - gain_db (float, default 20.0)
      Amount of gain, in decibels, applied during the mastering stage to achieve a target loudness. Adjust to taste for final output level.

    - sf2_name (str, default 'SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2')
      Filename of the SoundFont stored under a "models" directory in the current working directory. The function will construct models/{sf2_name} and read its bytes.

    - return_audio (bool, default False)
      If True the function returns the stereo numpy array (shape: [channels, samples] or [samples, channels] depending on master_mono_piano output convention). If False the function returns None.

    Returns
    - stereo (numpy.ndarray) or None
      When return_audio is True, returns the mastered stereo audio buffer. When False, nothing is returned.

    Side effects
    - Reads models/{sf2_name} from the current working directory.
    - Reads input_midi_file from disk.
    - Writes output_audio_file to disk using soundfile (sf.write).
    - Prints progress messages to stdout at major steps for diagnostic visibility.

    Dependencies and expectations
    - Requires the following components or equivalents in scope:
      - EncoderDecoder class with encode(waveform) and decode(latent, denoising_steps) methods.
      - A midirenderer.render_wave_from(sf2_bytes, midi_bytes) function that returns raw WAV bytes.
      - librosa for loading WAV bytes into numpy arrays.
      - master_mono_piano(waveform, gain_db) which returns (stereo_audio, diagnostics).
      - soundfile (imported as sf) for writing the final audio file.
    - The function assumes the rendered WAV is a mono waveform compatible with the encoder.
    - The models directory is expected at os.path.join(os.getcwd(), "models"). Ensure the process has read access to the SF2 file and write permission for the output path.

    Errors and exceptions
    - FileNotFoundError or OSError may be raised when reading files or when the SF2 or MIDI path is invalid.
    - Exceptions raised by EncoderDecoder, midirenderer, librosa, master_mono_piano, or sf.write propagate to the caller.
    - The caller should validate paths and catch exceptions for robust production use.

    Best practices and tips
    - For deterministic, reproducible output, fix seeds where the EncoderDecoder or decoder uses randomness and document those seeds externally.
    - To tune perceived noise/artifacts, experiment with denoising_steps in small increments and re-run quick tests on short MIDI files.
    - If rendering multi-instrumental MIDI, enable load_multi_instrumental_model only if the neural model and pipeline are designed for multiple instruments.
    - Monitor diagnostic output returned by master_mono_piano to drive automated quality checks or level adjustments.

    Example
    - Simple usage that saves to disk without returning audio:
      render_midi("input.mid", "final.wav")

    - Usage that returns the stereo audio for further programmatic processing:
      stereo = render_midi("input.mid", "final.wav", return_audio=True)
    """
    
    home_root = os.getcwd()
    models_dir = os.path.join(home_root, "models")
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
    stereo, diag = master_mono_piano(wv_rec, gain_db=gain_db)
    
    print('Saving final audio...')
    sf.write(output_audio_file, stereo.squeeze().T, samplerate=sr)
    
    print('=' * 70)
    print('Done!')
    print('=' * 70)
    
    if return_audio:
        return stereo