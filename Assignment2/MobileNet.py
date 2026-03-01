import torch.nn as nn
import torch


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = in_channels,
                kernel_size = 3,
                stride = stride,
                padding = 1,
                groups = in_chanels,
                bias = False
            ),
            nn.BatchNorm2d(in_channels),
            nnReLU6(inplace = True)
        )

        self.pointwise_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 1,
                bias = False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        intermediate = self.depthwise_convolution(x)
        return self.pointwise_convolution(intermediate)

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()

        self.residual_will_be_used = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expand_ratio)

        list_of_layers = []

        if expand_ratio != 1:
            list_of_layers.append(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            )
            list_of_layers.append(nn.BatchNorm2d(hidden_dim))
            list_of_layers.append(nn.ReLU6(inplace=True))

        list_of_layers.append(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size = 3,
                stride = stride,
                padding = 1,
                groups = hidden_dim,
                bias = False
            )
        )
        list_of_layers.append(nn.BatchNorm2d(hidden_dim))
        list_of_layers.append(nn.ReLU6(inplace=True))

        list_of_layers.append(
            nn.Conv2d(hidden_dim, out_channels, kernel_size = 1, bias=False)
        )
        list_of_layers.append(nn.BatchNorm2d(out_channels))
        self.sequential = nn.Sequential(*list_of_layers)

    def forward(self, x):
        if self.residual_will_be_used:
            return x + self.sequential(x)
        return self.sequential(x)
        

class MobileNet(nn.Module):
    def __init__(self, num_classes=18, width_mult=1.0, dropout_prob=0.2):
        super(MobileNet, self).__init__()

        number_of_output_channels_in_initial_convolution = int(32 * width_mult)

        self.initial_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels=number_of_output_channels_in_initial_convolution,
                kernel_size = 3,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(number_of_output_channels_in_initial_convolution),
            nn.ReLU6(inplace=True)
        )

        list_of_layers = []

        configuration = [
            (32, 16, 1, 1),
            (16, 24, 2, 6),
            (24, 24, 1, 6),
            (24, 32, 2, 6),
            (32, 32, 1, 6),
            (32, 32, 1, 6),
            (32, 64, 2, 6),
            (64, 64, 1, 6),
            (64, 64, 1, 6),
            (64, 64, 1, 6),
            (64, 96, 1, 6),
            (96, 96, 1, 6),
            (96, 96, 1, 6),
            (96, 160, 2, 6),
            (160, 160, 1, 6),
            (160, 160, 1, 6),
            (160, 320, 1, 6)
        ]

        for in_c, out_c, stride, expand_ratio in configuration:
            scaled_number_of_input_channels = int(in_c * width_mult)
            scaled_number_of_output_channels = int(out_c * width_mult)

            list_of_layers.append(
                InvertedResidual(
                    in_channels = scaled_number_of_input_channels,
                    out_channels = scaled_number_of_output_channels,
                    stride = stride,
                    expand_ratio = expand_ratio
                )
            )

        self.sequential = nn.Sequential(*list_of_layers)
        number_of_output_channels_in_final_convolution = int(1280 * width_mult) if width_mult > 1.0 else 1280

        self.final_convolution = nn.Sequential(
            nn.Conv2d(
                in_channels = int(320 * width_mult),
                out_channels = number_of_output_channels_in_final_convolution,
                kernel_size = 1,
                bias = False
            ),
            nn.BatchNorm2d(number_of_output_channels_in_final_convolution),
            nn.ReLU6(inplace = True)
        )

        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear_transformation = nn.Linear(
            number_of_output_channels_in_final_convolution,
            num_classes
        )

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        intermediate = self.initial_convolution(x)
        intermediate = self.sequential(intermediate)
        intermediate = self.final_convolution(intermediate)
        intermediate = self.average_pooling(intermediate)
        intermediate = torch.flatten(intermediate, 1)
        intermediate = self.dropout(intermediate)
        return self.linear_transformation(intermediate)