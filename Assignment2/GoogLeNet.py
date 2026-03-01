import torch.nn as nn
import torch


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()

        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size = 1, bias = False),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size = 1, bias = False),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(inplace = True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace = True)
        )

        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size = 1, bias = False),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(inplace = True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size = 5, padding = 2, bias = False),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace = True)
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            nn.Conv2d(in_channels, pool_proj, kernel_size = 1, bias = False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace = True)
        )


    def forward(self, x):

        output_1 = self.branch_1(x)
        output_2 = self.branch_2(x)
        output_3 = self.branch_3(x)
        output_4 = self.branch_4(x)

        output_list = [output_1, output_2, output_3, output_4]

        return torch.cat(output_list, dim=1)


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=18):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        self.inception3a = InceptionBlock(64, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(480, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x