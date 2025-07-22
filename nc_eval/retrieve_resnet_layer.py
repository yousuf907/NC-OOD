import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18

def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module

def get_activation(arch, all_outputs, name):
    def hook(model, input, output):
        
        if arch == 'ResNet18' and name == "layer4.1.conv2":
            all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(output, 1).squeeze(), 1).detach() # NXCXHXW >> N x C x 1 x 1 >> N x C
        elif arch == 'ResNet34' and name == "layer4.2.conv2":
            all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(output, 1).squeeze(), 1).detach() # NXCXHXW >> N x C x 1 x 1 >> N x C
        elif len(output.shape) == 2:
            all_outputs[name] = output.detach()
        elif len(output.shape) > 2 and output.shape[2] > 2:
            all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(output, 2).squeeze(), 1).detach() # NXCXHXW >> N x C x 2 x 2 >> N x 4C
            ##all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(output, 1).squeeze(), 1).detach() # NXCXHXW >> N x C x 1 x 1 >> N x C
        else:
            all_outputs[name] = torch.flatten(output.squeeze(), 1).detach()

    return hook


def add_hooks(arch, model, outputs, output_layer_names):
    """
    :param model:
    :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
    :param output_layer_names:
    :return:
    """
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(get_activation(arch, outputs, output_layer_name))


class ModelWrapper(nn.Module):
    def __init__(self, arch, model, output_layer_names, return_single=False):
        super(ModelWrapper, self).__init__()
        self.arch = arch
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.arch, self.model, self.outputs, self.output_layer_names)

    def forward(self, images):
        self.model(images)
        output_vals = {}
        #output_vals = [self.outputs[output_layer_name] for output_layer_name in self.output_layer_names]
        for name in self.output_layer_names:
            #if int(name[9:]) < 10:
            #    output_vals[name.replace('.', '0')] = self.outputs[name]
            #else:
            output_vals[name.replace('.', '')] = self.outputs[name]

        #if self.return_single:
        #    return output_vals[0]
        #else:
        #    return output_vals
        return output_vals


def test_resnet18():
    output_layer_names = ['layer1.0.bn1', 'layer4.0', 'fc']
    in_tensor = torch.ones((2, 3, 224, 224))

    core_model = resnet18()
    wrapper = ModelWrapper(core_model, output_layer_names)
    y1, y2, y3 = wrapper(in_tensor)
    assert y1.shape[0] == 2
    assert y1.shape[2] == 56
    assert y2.shape[2] == 7
    assert y3.shape[1] == 1000


if __name__ == "__main__":
    test_resnet18()
