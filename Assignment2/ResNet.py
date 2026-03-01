import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.convolution_1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = stride,
            padding = 1,
            bias = False
        )

        self.batch_normalization_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

        self.convolution_2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False
        )

        self.batch_normalization_2 = nn.BatchNorm2d(out_channels)

        shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 1,
                    stride = stride,
                    bias = False
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.shortcut = shortcut

    def forward(self, x):
        intermediate = self.convolution_1(x)
        intermediate = self.batch_normalization_1(intermediate)
        intermediate = self.relu(intermediate)
        intermediate = self.convolution_2(intermediate)
        intermediate = self.batch_normalization_2(intermediate)
        intermediate = intermediate + self.shortcut(x)

        return self.relu(intermediate)




class ResNet(nn.Module):
    def __init__(self, num_classes=18):
        super(ResNet, self).__init__()
        
        self.convolution = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 64,
                kernel_size = 7,
                stride = 2,
                padding = 3,
                bias = False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )

        self.max_pooling = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.residual_layer_1 = self._make_layer(64, 64, num_blocks = 2, stride = 1)
        self.residual_layer_2 = self._make_layer(64, 128, num_blocks = 2, stride = 2)
        self.residual_layer_3 = self._make_layer(128, 256, num_blocks = 2, stride = 2)
        self.residual_layer_4 = self._make_layer(256, 512, num_blocks = 2, stride = 2)

        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_transformation = nn.Linear(512, num_classes)


    def _make_layer(self, in_channels, out_channels, num_blocks, stride):

        list_of_blocks = []

        list_of_blocks.append(BasicBlock(in_channels, out_channels, stride = stride))

        for _ in range(1, num_blocks):
            list_of_blocks.append(BasicBlock(out_channels, out_channels, stride = 1))

        return nn.Sequential(*list_of_blocks)



    def forward(self, x):        
        intermediate = self.convolution(x)
        intermediate = self.max_pooling(intermediate)
        intermediate = self.residual_layer_1(intermediate)
        intermediate = self.residual_layer_2(intermediate)
        intermediate = self.residual_layer_3(intermediate)
        intermediate = self.residual_layer_4(intermediate)
        intermediate = self.average_pooling(intermediate)
        intermediate = torch.flatten(intermediate, 1)
        return self.linear_transformation(intermediate)