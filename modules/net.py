import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_tanh(x):
    return torch.clamp(x, min=-1, max=1)


class RichPoorTextureContrastModel(nn.Module):
    def __init__(self):
        
        super(RichPoorTextureContrastModel, self).__init__()
        self.hard_tanh = hard_tanh

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
        )
        self.conv_blocks0 = self._make_conv_block(32, 1)
        self.conv_blocks1 = self._make_conv_block(32, 3)
        self.conv_blocks2 = self._make_conv_block(32, 4)
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv_blocks3 = self._make_conv_block(32, 2)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv_blocks4 = self._make_conv_block(32, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 1)

    def _make_conv_block(self, out_channels, num_conv_layers):
        layers = []
        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, l1, l2):
        l1 = self.hard_tanh(self.feature_extraction(l1))
        l2 = self.hard_tanh(self.feature_extraction(l2))
    
        contrast = torch.abs(l1 - l2)
        x = self.conv_blocks0(contrast)
        # 连续的 3 次卷积
        x = self.conv_blocks1(x)

        x = self.conv_blocks2(x)
        x = self.pool1(x)

        x = self.conv_blocks3(x)
        x = self.pool2(x)

        x = self.conv_blocks4(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = torch.sigmoid(self.fc(x))
        return x
def get_model():
    return RichPoorTextureContrastModel()
def get_loss_fn():
    return nn.BCELoss()
def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)