import os

home_root = os.getcwd()

load_path_inference_multi_instrumental_default = os.path.join(home_root, 'models/music2latent.pt')
load_path_inference_solo_piano_default = os.path.join(home_root, 'models/music2latent_maestro_loss_16.871_iters_45500.pt')
load_path_inference_solo_piano_v1_default = os.path.join(home_root, 'models/music2latent_maestro_loss_27.834_iters_14300.pt')

max_batch_size_encode = 1                            # maximum inference batch size for encoding: tune it depending on the available GPU memory  
max_waveform_length_encode = 44100*60                # maximum length of waveforms in the batch for encoding: tune it depending on the available GPU memory
max_batch_size_decode = 1                            # maximum inference batch size for decoding: tune it depending on the available GPU memory
max_waveform_length_decode = 44100*60                # maximum length of waveforms in the batch for decoding: tune it depending on the available GPU memory

sigma_rescale = 0.06                                 # rescale sigma for inference