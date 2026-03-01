import torch.nn as nn
import torch


class VGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, num_convs=2):
        super(VGGBlock, self).__init__()
        in_channels_to_be_modified = in_channels

        layers = []

        for _ in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels = in_channels_to_be_modified,
                    out_channels = out_channels,
                    kernel_size = 3,
                    padding = 1,
                    bias = False
                )
            )
            
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace = True))
            in_channels_to_be_modified = out_channels
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class VGGNet(nn.Module):

    def __init__(self, num_classes=18):
        super(VGGNet, self).__init__()

        layers = []

        layers.append(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 64,
                kernel_size = 3,
                padding = 1,
                bias = False
            )
        )
        
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(VGGBlock(64, 128))
        layers.append(VGGBlock(128, 256))
        self.sequential = nn.Sequential(*layers)

        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.linear_transformation = nn.Linear(256, num_classes)
        

    def forward(self, x):
        intermediate = self.sequential(x)
        intermediate = self.average_pooling(intermediate)
        intermediate = torch.flatten(intermediate, 1)
        return self.linear_transformation(intermediate)