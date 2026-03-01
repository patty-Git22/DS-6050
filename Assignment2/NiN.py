import torch.nn as nn
import torch


class NiNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NiNBlock, self).__init__()

        layers = []

        layers.append(
            nn.Conv2d(
               in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
                bias = False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = 1,
                bias = False
            )
        ) 
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = 1,
                bias = False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class NiN(nn.Module):
    def __init__(self, num_classes=18):
        super(NiN, self).__init__()

        layers = []
        
        layers.append(NiNBlock(3, 96, kernel_size = 11, stride = 4))
        layers.append(nn.MaxPool2d(kernel_size = 3, stride = 2))
        layers.append(NiNBlock(96, 256, kernel_size =5 , padding = 2))
        layers.append(nn.MaxPool2d(kernel_size = 3, stride = 2))
        layers.append(NiNBlock(256, 384, kernel_size = 3, padding = 1))
        layers.append(nn.MaxPool2d(kernel_size = 3, stride = 2))
        layers.append(NiNBlock(384, num_classes, kernel_size = 3, padding = 1))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        intermediate = self.sequential(x)
        return torch.flatten(intermediate, 1)