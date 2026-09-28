import abc
from typing import Optional
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet34_Weights
import timm

class ConvRelu(nn.Module):
    """ Convolution -> ReLU.

        Args:   
            in_channels : number of input channels
            out_channels : number of output channels
            kernel_size : size of convolution kernel
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3) -> None:
        super(ConvRelu, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

class LocModel(nn.Module, metaclass=abc.ABCMeta):
    """ Base class for all localization models."""

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.forward_once(x))

    def _initialize_weights(self) -> None:
        """ Initialize weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Res34_Unet_Loc(LocModel):
    """Unet model with a resnet34 encoder used for localization. 
        
        Args:
            pretrained : if True, use pretrained resnet34 weights.

    """

    def __init__(self, pretrained: bool=True, **kwargs) -> None:
        super().__init__()
        encoder_filters = [64, 64, 128, 256, 512]
        decoder_filters = np.asarray([48, 64, 96, 160, 320])
        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)
        self._initialize_weights()
        if not isinstance(pretrained, bool):
            raise TypeError('pretrained argument should be a bool')
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        if weights is not None:
            print(f'using weights from {weights}')
        encoder = torchvision.models.resnet34(weights=weights)
        self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.conv2 = nn.Sequential(encoder.maxpool, encoder.layer1)
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], 1))
        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], 1))
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], 1))
        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))
        dec9 = self.conv9_2(torch.cat([dec9, enc1], 1))
        dec10 = self.conv10(F.interpolate(dec9, scale_factor=2))
        return dec10

class Siamese(nn.Module, metaclass=abc.ABCMeta):
    """ Abstract class for siamese networks. To create a 
    siamese network, inherit from this class and the class of the localization model.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        decoder_filters = np.asarray([48, 64, 96, 160, 320])
        self.res = nn.Conv2d(decoder_filters[-5] * 2, 5, 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass  for two inputs (that have been concateneted chanelwise to one). """
        output1 = self.forward_once(x[:, :3, :, :])
        output2 = self.forward_once(x[:, 3:, :, :])
        return self.res(torch.cat([output1, output2], 1))

class Res34_Unet_Double(Siamese, Res34_Unet_Loc):
    """ ResNet34 Unet model for classification tasks."""

    def encode_once(self, x: torch.Tensor) -> torch.Tensor:
        """ Encode one image with the encoder part of the model."""
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        return enc5

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """ Get the embeddings of the images."""
        encoded1 = self.encode_once(x[:, :3, :, :])
        encoded2 = self.encode_once(x[:, 3:, :, :])
        encoded = torch.cat([encoded1, encoded2], 1)
        return F.adaptive_avg_pool2d(encoded, 1).view(encoded.shape[0], -1)

class ResNeSt50_Unet_Loc(LocModel):

    def __init__(self, pretrained: bool=True, ckpt_path: Optional[str]=None, **kwargs) -> None:
        super().__init__()
        encoder_filters = [64, 256, 512, 1024, 2048]
        decoder_filters = np.asarray([48, 64, 96, 160, 320])
        self.conv6 = ConvRelu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvRelu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvRelu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvRelu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvRelu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvRelu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvRelu(decoder_filters[-3], decoder_filters[-4])
        self.conv9_2 = ConvRelu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        self.conv10 = ConvRelu(decoder_filters[-4], decoder_filters[-5])
        self.res = nn.Conv2d(decoder_filters[-5], 1, 1, stride=1, padding=0)
        self._initialize_weights()
        self.encoder = timm.create_model('resnest50d', pretrained=pretrained and ckpt_path is None, features_only=True, out_indices=(0, 1, 2, 3, 4))
        if ckpt_path is not None:
            print(f'Loading pretrained ResNeSt50 weights from {ckpt_path} ...')
            state_dict = torch.load(ckpt_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                if new_k.startswith('module.'):
                    new_k = new_k[len('module.'):]
                if new_k.startswith('model.'):
                    new_k = new_k[len('model.'):]
                new_state_dict[new_k] = v
            incompatible_keys = self.encoder.load_state_dict(new_state_dict, strict=False)
            print('Missing keys during weight loading:')
            print(incompatible_keys.missing_keys)
            print('Unexpected keys during weight loading:')
            print(incompatible_keys.unexpected_keys)
            bad_missing = [k for k in incompatible_keys.missing_keys if not k.startswith('fc') and (not k.startswith('classifier')) and (not k.startswith('head')) and (not k.startswith('global_pool'))]
            assert len(bad_missing) == 0, f'Backbone weights not loaded correctly. First missing keys: {bad_missing[:20]}'

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        enc1, enc2, enc3, enc4, enc5 = (features[0], features[1], features[2], features[3], features[4])
        dec6 = self.conv6(F.interpolate(enc5, size=enc4.shape[-2:], mode='nearest'))
        dec6 = self.conv6_2(torch.cat([dec6, enc4], dim=1))
        dec7 = self.conv7(F.interpolate(dec6, size=enc3.shape[-2:], mode='nearest'))
        dec7 = self.conv7_2(torch.cat([dec7, enc3], dim=1))
        dec8 = self.conv8(F.interpolate(dec7, size=enc2.shape[-2:], mode='nearest'))
        dec8 = self.conv8_2(torch.cat([dec8, enc2], dim=1))
        dec9 = self.conv9(F.interpolate(dec8, size=enc1.shape[-2:], mode='nearest'))
        dec9 = self.conv9_2(torch.cat([dec9, enc1], dim=1))
        dec10 = self.conv10(F.interpolate(dec9, size=x.shape[-2:], mode='nearest'))
        return dec10

class ResNeSt50_Unet_Double(Siamese, ResNeSt50_Unet_Loc):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        nn.init.kaiming_normal_(self.res.weight)
        if self.res.bias is not None:
            nn.init.zeros_(self.res.bias)

    def encode_once(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return features[-1]

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        encoded1 = self.encode_once(x[:, :3, :, :])
        encoded2 = self.encode_once(x[:, 3:, :, :])
        encoded = torch.cat([encoded1, encoded2], dim=1)
        return F.adaptive_avg_pool2d(encoded, 1).view(encoded.shape[0], -1)
