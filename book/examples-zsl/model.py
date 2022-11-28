import torch
import torch.nn as nn
import torchvision

class WordAudioSiameseNetwork(nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()
        
        self.audio_model = MelCNN(128)
        self.audio_projection = nn.Linear(in_features=128, out_features=128, bias=True)
        self.word_projection = nn.Linear(in_features=300, out_features=128, bias=True)
        
    def forward(self, x_audio, pos_word, neg_word):
        x_audio = self.audio_model(x_audio)
        x_audio = torch.squeeze(x_audio, dim=-1)
        x_audio = torch.squeeze(x_audio, dim=-1)
        x_audio = nn.Sigmoid()(self.audio_projection(x_audio))
        
        x_word_pos = nn.Sigmoid()(self.word_projection(pos_word))
        x_word_neg = nn.Sigmoid()(self.word_projection(neg_word))
        
        return x_audio, x_word_pos, x_word_neg

class ImageAudioSiameseNetwork(nn.Module):
    def __init__(
        self
    ) -> None:
        super().__init__()

        self.audio_model = MelCNN(128)
        self.audio_projection = nn.Linear(in_features=128, out_features=128, bias=True)

        # Using a pretrained resnet101 image classification model as a backbone.
        visual_model = torchvision.models.resnet101(pretrained=True)
        layers = list(visual_model.children())
        self.visual_model = nn.Sequential(*layers[:-1])
        for _m in self.visual_model.children():
            for param in _m.parameters():
                param.requires_grad = False
        self.visual_projection = nn.Linear(in_features=2048, out_features=128, bias=True)
                
    def forward(self, x_audio, pos_img, neg_img):
        x_audio = nn.Sigmoid()(self.audio_model(x_audio))
        x_audio = torch.squeeze(x_audio, dim=-1)
        x_audio = torch.squeeze(x_audio, dim=-1)
        x_audio = self.audio_projection(x_audio)

        pos_img = nn.Sigmoid()(self.visual_model(pos_img))
        pos_img = torch.squeeze(pos_img, dim=-1)
        pos_img = torch.squeeze(pos_img, dim=-1)
        x_img_pos = self.visual_projection(pos_img)
        
        neg_img = nn.Sigmoid()(self.visual_model(neg_img))
        neg_img = torch.squeeze(neg_img, dim=-1)
        neg_img = torch.squeeze(neg_img, dim=-1)
        x_img_neg = self.visual_projection(neg_img)
        
        return x_audio, x_img_pos, x_img_neg

    
class MelCNN(nn.Module):
    def __init__(self, emb_dim):
        super(MelCNN, self).__init__()

        # Spectrogram
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN : input (1, 63 * N, 80) / kernel size (3x3)
        self.layer1 = Conv_2d(1, 64, pooling=(1,2))
        self.layer2 = Conv_2d(64, 128, pooling=(3,4))
        self.layer3 = Conv_2d(128, 128, pooling=(7,5))
        self.layer4 = Conv_2d(128, 128, pooling=(3,2))
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.spec_bn(x)
        x = self.layer1(x)        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x
    
class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out