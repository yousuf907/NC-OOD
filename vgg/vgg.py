"""VGG."""
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init

## 224x224 resolutions ## max-pool in all 5 stages
cfg = {
    "VGG11": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG17": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
              512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, vgg_name, class_num=100, output_dim=0, hidden_mlp=0):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])

        # Projector
        if output_dim == 0:
            self.projection_head = None
            self.classifier = nn.Linear(512, class_num)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(512, hidden_mlp), ## 512 > 2048
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim), # 2048 > 512
            )
            self.classifier = nn.Linear(output_dim, class_num)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        nn.init.xavier_uniform_(self.classifier.weight)


    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                nn.init.kaiming_normal_(conv2d.weight, mode="fan_out", nonlinearity="relu")
                batchnorm = nn.BatchNorm2d(x)
                nn.init.constant_(batchnorm.weight, 1)
                nn.init.constant_(batchnorm.bias, 0)

                layers += [
                    conv2d,
                    batchnorm,
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
            xe=xe0

        x_out = self.classifier(xe)

        if embed:
            return xe0, xe, x_out
        else:
            return x_out


if __name__ == "__main__":
    from types import SimpleNamespace

    #model = VGG("VGG11", class_num=100)
    model = VGG("VGG17", class_num=100)
    #print(model)
    x = torch.randn((10, 3, 224, 224), dtype=torch.float32)
    y = model(x)
    print('Shape of y:', y.shape)

    n_parameters = sum(p.numel() for p in model.parameters())
    print('\nNumber of Params (in Millions):', n_parameters / 1e6)

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
