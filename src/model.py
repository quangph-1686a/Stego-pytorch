import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, kernel_size=3, channel=3, filters=50):
        super(ConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.channel = channel
        self.filters = filters
        self.pd = (self.kernel_size - 1) // 2
        self.conv0 = nn.Conv2d(self.channel, self.filters, self.kernel_size,
                               stride=1, padding=self.pd)
        self.conv1 = nn.Conv2d(self.filters, self.filters, self.kernel_size,
                               stride=1, padding=self.pd)
        self.conv2 = nn.Conv2d(self.filters, self.filters, self.kernel_size,
                               stride=1, padding=self.pd)
        self.conv3 = nn.Conv2d(self.filters, self.filters, self.kernel_size,
                               stride=1, padding=self.pd)
        self.conv4 = nn.Conv2d(self.filters, self.filters, self.kernel_size,
                               stride=1, padding=self.pd)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = F.relu(self.conv4(x))

        return out


class BaseModel(nn.Module):

    def __init__(self, channel=3):
        super(BaseModel, self).__init__()
        self.kernel_sizes = [3, 5, 3]
        self.channel = channel
        self.convblock0 = ConvBlock(kernel_size=self.kernel_sizes[0], channel=self.channel)
        self.convblock1 = ConvBlock(kernel_size=self.kernel_sizes[1], channel=self.channel)
        self.convblock2 = ConvBlock(kernel_size=self.kernel_sizes[2], channel=self.channel)

    def forward(self, x):
        out = torch.cat(
                    [self.convblock0(x),
                     self.convblock1(x),
                     self.convblock2(x)],
                    axis=1)
        return out


class DecodeImageModel(nn.Module):

    def __init__(self):
        super(DecodeImageModel, self).__init__()
        self.filters = 50
        self.preparenet = BaseModel(channel=2)
        self.hidingnet = BaseModel(channel=self.filters * 3 + 3)
        self.conv = nn.Conv2d(self.filters * 3, 3, kernel_size=1,
                              stride=1, padding=0)

    def forward(self, image_in, audio_in):
        x = image_in
        y = self.preparenet(audio_in)
        xy = torch.cat((x, y), axis=1)
        xy = self.hidingnet(xy)
        out = F.relu(self.conv(xy))

        return out


class DecodeAudioModel(nn.Module):

    def __init__(self):
        super(DecodeAudioModel, self).__init__()
        self.channel = 2
        self.filters = 50
        self.revealingnet = BaseModel(channel=3)
        self.conv = nn.Conv2d(self.filters * 3, self.channel, kernel_size=1,
                              stride=1, padding=0)
        #self,linear = nn.Linear()

    def forward(self, image_in):
        x = self.revealingnet(image_in)
        out = (self.conv(x))
        return out


class EncodeDecodeModel(nn.Module):

    def __init__(self):
        super(EncodeDecodeModel, self).__init__()
        self.encoder = DecodeImageModel()
        self.decoder = DecodeAudioModel()

    def forward(self, image_in, audio_in):
        cover_img = self.encoder(image_in, audio_in)
        revealing_audio = self.decoder(cover_img)
        return cover_img, revealing_audio


# encode_decoder_model = EncodeDecodeModel()
# print(encode_decoder_model)
