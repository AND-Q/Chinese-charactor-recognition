from torch import nn
import Residual_block
import vgg_block
from d2l import torch as d2l
from torchvision.models import resnet18

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # 16*64*64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16*32*32
            nn.Conv2d(16, 32, 3, stride=1, padding=0),  # 32*30*30
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # 32*15*15

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(32 * 15 * 15, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 3755))

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.layer1 = nn.Sequential(  # bs*1*64*64
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 64*32*32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64*16*16   (32+2-3/2)+1
        )
        # 第2层64*16*16(2个resnet_block，4个卷积层）
        self.layer2 = nn.Sequential(*Residual_block.resnet_block(64, 64, 2, first_block=True))
        # 第3层通道数翻倍，高宽减半128*8*8（2个resnet_block，4个卷积层）
        self.layer3 = nn.Sequential(*Residual_block.resnet_block(64, 128, 2))
        # 第4层通道数翻倍，高宽减半256*4*4（2个resnet_block，4个卷积层）
        self.layer4 = nn.Sequential(*Residual_block.resnet_block(128, 256, 2))
        # 第5层通道数翻倍，高宽减半512*2*2（2个resnet_block，4个卷积层）
        self.layer5 = nn.Sequential(*Residual_block.resnet_block(256, 512, 2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 平均池化512*1*1
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 3755)  # 512->3755


        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.layer5(X)
        X = self.avgpool(X)
        X = self.flatten(X)
        net = self.fc(X)
        return net

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=None)  # 或 weights='DEFAULT' 使用预训练
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 单通道适配
        self.model.fc = nn.Linear(512, 3755)

    def forward(self, x):
        return self.model(x)

from torchvision.models import resnet34
import torch.nn as nn

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = resnet34(weights=None)  # 如果用预训练，可写 weights='DEFAULT'
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 单通道适配
        self.model.fc = nn.Linear(512, 3755)  # 修改输出类别数

    def forward(self, x):
        return self.model(x)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # 配置：每个元组是 (卷积层数量, 输出通道数)
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        layers = []
        in_channels = 1
        # 1*64*64->64*32*32->128*16*16->256*8*8->512*4*4->512*2*2
        # 1*128*128->64*64*64->128*32*32->256*16*16->512*8*8->512*4*4
        for (num_convs, out_channels) in conv_arch:
            layers.append(vgg_block.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)  # 卷积层部分

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(  # 全连接层部分
            # nn.Linear(512 * 2 * 2, 4096),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(4096, 3755)  # 3755分类

            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 3755)  # 3755分类
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    # def forward(self, x):
    #     print("Input shape:", x.shape)
    #     for i, layer in enumerate(self.features):
    #         x = layer(x)
    #         print(f"Features[{i}] output shape: {x.shape}")
    #     x = self.flatten(x)
    #     print("Flatten shape:", x.shape)
    #     x = self.classifier(x)
    #     return x


class LeNet(nn.Module):  # 不稳定
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(  # bs*1*64*64
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),  # 6*64*64（sigmoid->relu）
            nn.AvgPool2d(kernel_size=2, stride=2),  # 6*32*32
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),  # 16*28*28
            nn.AvgPool2d(kernel_size=2, stride=2),  # 16*14*14
            nn.Flatten(),
            # nn.Linear(16 * 14 * 14, 120), nn.ReLU(),#3136->120
            # nn.Linear(120, 84), nn.ReLU(),#120->84
            # nn.Linear(84, 3755))#84->3755
            nn.Linear(16 * 14 * 14, 1024), nn.ReLU(),  # 从 3136 → 1024，减缓降维
            nn.Dropout(0.5),  # 不加会出现严重的过拟合（训练到30epoch）acc为0.1左右！
            nn.Linear(1024, 512), nn.ReLU(),  # 1024 → 512
            nn.Dropout(0.5),
            nn.Linear(512, 3755))

    def forward(self, x):
        out = self.net(x)
        return out
