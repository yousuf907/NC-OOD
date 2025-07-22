import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import *

def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module

def get_activation(all_outputs, name):
    def hook(model, input, output):
        #all_outputs[name] = output.detach()
        #all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(output, 2).squeeze(), 1).detach()
        #print(output.shape)
        #print(name)

        if name == "features.49":
            #gn = nn.GroupNorm(32, 512).cuda()
            #relu= nn.ReLU(inplace=True).cuda()
            #mp=nn.MaxPool2d(kernel_size=2, stride=2).cuda()
            #all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(mp(relu(gn(output))), 1).squeeze(), 1).detach()
            all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(output, 1).squeeze(), 1).detach() # NXCXHXW >> NxCx1x1 >> NxC
        #if name == "module.features.49":
        #    all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(output, 1).squeeze(), 1).detach() ##>>  NXCXHXW >> NxCx1x1 >> NxC
        elif len(output.shape) == 2:
            all_outputs[name] = output.detach()
        elif len(output.shape) > 2 and output.shape[2] > 2:
            all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(output, 2).squeeze(), 1).detach() # NXCXHXW >> N x C x 2 x 2 >> N x 4C
            ##all_outputs[name] = torch.flatten(F.adaptive_avg_pool2d(output, 1).squeeze(), 1).detach() # NXCXHXW >> N x C x 1 x 1 >> N x C
        else:
            all_outputs[name] = torch.flatten(output.squeeze(), 1).detach()

    return hook


def add_hooks(model, outputs, output_layer_names):
    """
    :param model:
    :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
    :param output_layer_names:
    :return:
    """
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(get_activation(outputs, output_layer_name))


class ModelWrapper(nn.Module):
    def __init__(self, model, output_layer_names, return_single=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, images):
        self.model(images)
        output_vals = {}
        #output_vals = [self.outputs[output_layer_name] for output_layer_name in self.output_layer_names]
        for name in self.output_layer_names:
            #print(name)
            if "projection_head" in name:
                output_vals[name.replace('.', '')] = self.outputs[name]
            elif int(name[9:]) < 10:
                output_vals[name.replace('.', '0')] = self.outputs[name]
                #print(name)
            else:
                output_vals[name.replace('.', '')] = self.outputs[name]
                #print(name)

        #if self.return_single:
        #    return output_vals[0]
        #else:
        #    return output_vals
        return output_vals


def test_vgg11():
    
    output_layer_names = [
                        "features.0",
                        "features.3",
                        "features.6",
                        "features.9",
                        "features.12",
                        "features.15",
                        "features.19",
                        "features.22",
                        "features.26",
                        "features.29"
                    ]

    #in_tensor = torch.ones((2, 3, 224, 224))
    in_tensor = torch.ones((2, 3, 32, 32))

    core_model = VGG("VGG11", class_num=10)
    wrapper = ModelWrapper(core_model, output_layer_names)
    #y1,y2,y3,y4,y5,y6,y7,y8,y9,y10 = wrapper(in_tensor)
    y1 = wrapper(in_tensor)
    print(len(y1))
    for k in y1:
        print(k)
        print(y1[k].shape)
    #assert y1.shape[0] == 2
    #assert y1.shape[2] == 56
    #assert y2.shape[2] == 7
    #assert y3.shape[1] == 1000


if __name__ == "__main__":
    #test_resnet18()
    test_vgg11()
