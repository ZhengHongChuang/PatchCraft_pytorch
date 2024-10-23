import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_tanh(x):
    return torch.maximum(torch.minimum(x, torch.tensor(1.)), torch.tensor(-1.))


class RichPoorTextureContrastModel(nn.Module):

    def __init__(self):

        super(RichPoorTextureContrastModel, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
        )
        self.conv_blocks0 = self._make_conv_block(32, 32, 4)
        self.conv_blocks1 = self._make_conv_block(32, 32, 2)
        self.conv_blocks2 = self._make_conv_block(32, 32, 2)
        self.conv_blocks3 = self._make_conv_block(32, 32, 2)

        self.pool = nn.AvgPool2d(3)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def _make_conv_block(self, in_channels, out_channels, num_conv_layers):
        layers = []
        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, l1, l2):
        l1 = hard_tanh(self.feature_extraction(l1))
        l2 = hard_tanh(self.feature_extraction(l2))

        x = torch.subtract(l1, l2)
        x = self.conv_blocks0(x)
        x = self.pool(x)

        x = self.conv_blocks1(x)
        x = self.pool(x)

        x = self.conv_blocks2(x)
        x = self.pool(x)

        x = self.conv_blocks3(x)
        x = self.pool(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x





def get_model():
    return RichPoorTextureContrastModel()


def get_loss_fn():
    return nn.BCELoss()


def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


if __name__ == "__main__":
    model = RichPoorTextureContrastModel().to('cuda')
    from torchsummary import summary
    summary(model, input_size=[(1, 256, 256), (1, 256, 256)])
