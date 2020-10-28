import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.io.wavfile import write
from scipy.signal import istft
from utils import load_audio_stft
import torch

IMG_TEST = '../test_data/image/5.jpg'
AUDIO_TEST = '../test_data/audio/10.wav'
OUT_PATH = '../outputs/encode.png'
STEGO_MODEL = '../pre-trained2/encoder_40.pt'


def stego_inference(img_path, audio_path, output_path, model):

    image = Image.open(img_path).convert("RGB")
    audio = np.array(load_audio_stft(audio_path), dtype=np.float32)
    image = np.array(ImageOps.fit(image, (255, 255)), dtype=np.float32) / 255.

    image = np.transpose(image, (2, 0, 1))
    audio = np.transpose(audio, (2, 0, 1))
    
    image = torch.tensor(np.expand_dims(image, axis=0))
    audio = torch.tensor(np.expand_dims(audio, axis=0))
    
    stego_image = model(image, audio).detach().numpy()
    stego_image = np.transpose(np.squeeze(stego_image), (1, 2, 0))
    stego_image = (np.squeeze(stego_image) * 255).astype(np.uint8)

    plt.imsave(output_path, stego_image)

if __name__ == '__main__':
    stg_model = torch.load(STEGO_MODEL, map_location=torch.device('cpu'))
    stego_inference(IMG_TEST, AUDIO_TEST, OUT_PATH, stg_model)
