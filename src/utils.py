import numpy as np
from scipy.io.wavfile import read
from scipy.signal import stft


def load_audio_stft(path_audio_file, max_feature=64516):
    sr, s1 = read(path_audio_file)
    if len(s1) < max_feature:
        s1 = np.pad(s1, (0, max_feature - len(s1)),
                    mode='constant', constant_values=0)
    else:
        s1 = s1[:max_feature]
    s1 = s1 / (2 ** 10)
    freqs, bins, sxx = stft(s1, nperseg=254 * 2, fs=sr, noverlap=254)
    real_part = np.real(sxx)
    real_part = np.expand_dims(real_part, axis=-1)
    complex_part = np.imag(sxx)
    complex_part = np.expand_dims(complex_part, axis=-1)

    return np.concatenate((real_part, complex_part), axis=-1)
