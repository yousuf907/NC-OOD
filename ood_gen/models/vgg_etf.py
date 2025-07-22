"""VGG"""
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import numpy as np

class Conv2d(nn.Conv2d): # For Weight Standardization
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv2d_init(m):
    assert isinstance(m, nn.Conv2d)
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))

def gn_init(m, zero_init=False):
    assert isinstance(m, nn.GroupNorm)
    m.weight.data.fill_(0. if zero_init else 1.)
    m.bias.data.zero_()

## 224x224 resolutions ## max-pool in all 5 stages
cfg = {
    "VGG11": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG17": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
              512, 512, 512, 512, 'M']
}

## VGG17: VGG17 is created by removing last FC layers from original VGG19

class VGG(nn.Module):
    def __init__(self, vgg_name, class_num=100, output_dim=0, hidden_mlp=0):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

        # ETF Projector
        if output_dim == 0:
            self.projection_head = None
            self.classifier = nn.Linear(512, class_num)
        else:
            ## ETF (NC) >> Two Layers (512 --> 2048 --> 512)
            self.projection_head = nn.Sequential(
                nn.Linear(in_features=512, out_features=hidden_mlp, bias=False), ## 512 > 2048
                nn.ReLU(inplace=True),
                nn.Linear(in_features=hidden_mlp, out_features=output_dim, bias=False), # 2048 > 512
            )

            shape, c = self.projection_head[0].weight.shape, output_dim
            self.projection_head[0].weight = nn.Parameter(
                (c/(c-1))**(1/2) * (torch.eye(*shape) - (1/c) * torch.ones(*shape)), requires_grad=False)

            shape, c = self.projection_head[2].weight.shape, output_dim
            self.projection_head[2].weight = nn.Parameter(
                (c/(c-1))**(1/2) * (torch.eye(*shape) - (1/c) * torch.ones(*shape)), requires_grad=False)

            ## classifier head
            self.classifier = nn.Linear(output_dim, class_num)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        nn.init.xavier_uniform_(self.classifier.weight) ##>> activate for regular Cross-Entropy loss and not Cosine Loss

        ## Init GN and WS part
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv2d_init(m)
            elif isinstance(m, nn.GroupNorm):
                gn_init(m)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                ## GN+WS
                conv2d = conv3x3(in_channels, x)
                groupnorm = nn.GroupNorm(32, x)

                layers += [
                    conv2d,
                    groupnorm,
                    nn.ReLU(inplace=True),
                ]

                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


    def forward(self, x, embed=False):
        x = self.features(x)
        x = self.avgpool(x)
        xe0 = x.view(x.size(0), -1)

        if self.projection_head is not None:
            xe = self.projection_head(xe0)
            xe = F.normalize(xe, eps=1e-8, p=2, dim=-1)
        else:
            xe = xe0

        x_out = self.classifier(xe)

        if embed:
            return xe0, xe, x_out
        else:
            return x_out


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])



if __name__ == "__main__":
    from types import SimpleNamespace

    #model = VGG("VGG11", class_num=100)
    #model = VGG("VGG17", class_num=100)
    model = VGG("VGG17", class_num=100, output_dim=512, hidden_mlp=2048)
    #print(model)

    x = torch.randn((10, 3, 224, 224), dtype=torch.float32) # B x C x H x W
    y = model(x)
    print('\nShape of y:', y.shape)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    n_parameters = sum(p.numel() for p in model.parameters())
    print('\nNumber of Total Params (in Millions):', n_parameters / 1e6)

    p = count_network_parameters(model)
    print('\nNumber of Trainable Params (in Millions):', p / 1e6)
