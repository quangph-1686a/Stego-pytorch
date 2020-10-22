import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.io.wavfile import write
from scipy.signal import istft
from utils import load_audio_stft
import torch


IN_PATH = '../outputs/encode.png'
OUT_PATH = '../outputs/decode.wav'
REVEALING_MODEL = '../pre-trained/decoder.pt'


def revealing_inference(img_path, output_path, model):

    img = Image.open(img_path).convert("RGB")
    img = np.array(ImageOps.fit(img, (255, 255)), dtype=np.float32) / 255.
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img)
    revealing_audio = model(img).detach().numpy()
    revealing_audio = np.transpose(np.squeeze(revealing_audio), (1, 2, 0))

    revealing_audio *= (2 ** 10)
    revealing_audio = revealing_audio[:, :, 0] + revealing_audio[:, :, 1] * 1j

    t, x = istft(np.abs(revealing_audio) * np.exp(1j * np.angle(revealing_audio)),
                 fs=16000, nfft=254 * 2, nperseg=254 * 2, noverlap=254)
    x = x.astype(np.int16)

    write(output_path, 16000, x)

    return x


if __name__ == '__main__':
    revealing_model = torch.load(REVEALING_MODEL, map_location=torch.device('cpu'))
    revealing_inference(IN_PATH, OUT_PATH, revealing_model)
