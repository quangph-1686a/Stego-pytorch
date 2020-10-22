from utils import load_audio_stft
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import os


class StagoDataset(Dataset):

    def __init__(self, path_image_dir, path_audio_dir,
                 num_step_on_epoch=1000,
                 batch_size=8, size=(255, 255)):
        self.path_image_dir = path_image_dir
        self.path_audio_dir = path_audio_dir
        self.train_images = os.listdir(path_image_dir)
        self.train_audios = os.listdir(path_audio_dir)
        self.batch_size = batch_size
        self.image_size = size
        self.num_step_on_epoch = num_step_on_epoch

    def __len__(self):
        return self.num_step_on_epoch * self.batch_size # min(len(self.train_images), len(self.train_audios))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.path_image_dir,
                                random.choice(self.train_images))
        audio_path = os.path.join(self.path_audio_dir,
                                  random.choice(self.train_audios))

        image = Image.open(img_path).convert("RGB")

        image = np.array(ImageOps.fit(image, self.image_size),
                         dtype=np.float32) / 255.
        image = np.transpose(image, (2, 0, 1))
        audio = np.array(load_audio_stft(audio_path), dtype=np.float32)
        audio = np.transpose(audio, (2, 0, 1))
        sample = {'image': image, 'audio': audio}

        return sample
