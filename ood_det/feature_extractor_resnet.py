import argparse
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm.contrib import tqdm
#from models.model_utils import get_model
from data.datasets import load_dataset
from utils import checkpoint_summary
from retrieve_resnet_layer import ModelWrapper

from pathlib import Path

class Feature_Extractor:
    def __init__(self, cfg:argparse.Namespace, ckpt_dir: Path=None, feature_dir: Path=None, model=None) -> None:
        self.cfg = cfg
        self.dataset_name = cfg.set
        # get model
        assert ckpt_dir is not None or model is not None, "Either ckpt_dir or model should be active."
        self.feature_dir = feature_dir

        self.model = self.load_model(ckpt_dir, model)
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.model.cuda().eval()
        #self.hooks = self.attach_hooks()
        #self.keys = list(self.hooks.keys())

        # get data
        train_loader, val_loader, test_loader = load_dataset(self.cfg)
        self.loaders = {'train':train_loader, 'val':val_loader, 'test':test_loader}

        self.activations = {}
        # self._dummy_batch()

    def load_model(self, ckpt_dir, model=None):
        if model is not None:
            return model

        if self.cfg.nc_control:
            from models.model_utils_nc import get_model
        else:
            from models.model_utils import get_model

        # print(f"loading model from {ckpt_dir}")
        _model = get_model(self.cfg)
        # sys.exit()
        _model_dict = _model.state_dict()


        ckpt = torch.load(ckpt_dir)
        for k in ckpt.keys():
            if 'state_dict' in k:
                _state_dict_key = k

        ### Remove module
        state_dict = ckpt[_state_dict_key]
        for k in list(state_dict.keys()):
            # retain model only
            if k.startswith('module'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        ckpt[_state_dict_key] = state_dict


        #print("\nThis is the pre-trained ckpt:")
        #for param_tensor in ckpt[_state_dict_key]:
        #    print(param_tensor, "\t", ckpt[_state_dict_key][param_tensor].size())

        # if self.cfg.debug:
        #     print(f"checkpoint keys:\n{ckpt.keys()}")
        layers_to_drop = [k for k in ckpt[_state_dict_key].keys() if 'layer.' in k or 'mask' in k]
        #if 'transfer' in self.cfg.task:
        if self.cfg.arch.lower() == 'resnet18':
            layers_to_drop.extend(['fc.weight', 'fc.bias'])
        elif self.cfg.arch.lower() == 'resnet34':
            layers_to_drop.extend(['fc.weight', 'fc.bias'])
        elif self.cfg.arch.lower() == 'vgg11':
            layers_to_drop.extend(['classifier.weight', 'classifier.bias'])
        elif self.cfg.arch.lower() == 'vgg17':
            layers_to_drop.extend(['classifier.weight', 'classifier.bias'])

        for k in layers_to_drop:
            ckpt[_state_dict_key].pop(k)


        _model_dict.update(ckpt[_state_dict_key])

        _model.load_state_dict(_model_dict)

        model_state_dict = _model.state_dict()
        for k in ckpt[_state_dict_key]:
            #if not k.startswith('fc.'):
            assert torch.equal(model_state_dict[k].cpu(), ckpt[_state_dict_key][k].cpu()), k
        print("\nSuccessfully performed sanity-check on loading pre-trained ckpt")

        if 'best_val_acc' in ckpt:
            self.best_val_acc1 = ckpt['best_val_acc']
            if 'best_test_acc' in ckpt.keys():
                self.best_test_acc1 = ckpt['best_test_acc']

            checkpoint_summary(ckpt_dir, ckpt)
        # print(f"checkpoint's best accuracy is {ckpt['best_val_acc']}")
        return _model

    def attach_hooks(self):
        hooks = {}
        patch_size=2

        def get_name_to_module(model):
            name_to_module = {}
            for m in model.named_modules():
                print(m[0])
                name_to_module[m[0]] = m[1]
            return name_to_module

        def _getActivation(name):
            # the hook signature
            def hook(model, input, output):
                #self.activations[name.replace('.', '')] = torch.flatten(F.adaptive_avg_pool2d(input[0], 2).squeeze(), 1).detach()

                if self.cfg.arch == 'ResNet18' and name == "layer4.1.conv2":
                    self.activations[name.replace('.', '')] = torch.flatten(F.adaptive_avg_pool2d(input[0], 1).squeeze(), 1).detach() # NXCXHXW >> N x C x 1 x 1 >> N x C
                elif self.cfg.arch == 'ResNet34' and name == "layer4.2.conv2":
                    self.activations[name.replace('.', '')] = torch.flatten(F.adaptive_avg_pool2d(input[0], 1).squeeze(), 1).detach() # NXCXHXW >> N x C x 1 x 1 >> N x C
                elif len(input[0].shape) == 2:
                    self.activations[name.replace('.', '')] = input[0].detach()
                elif len(input[0].shape) > 2 and input[0].shape[2] > 2:
                    self.activations[name.replace('.', '')] = torch.flatten(F.adaptive_avg_pool2d(input[0], 2).squeeze(), 1).detach() # NXCXHXW >> N x C x 2 x 2 >> N x 4C
                else:
                    self.activations[name.replace('.', '')] = torch.flatten(input[0].squeeze(), 1).detach()

            return hook

        name_to_module = get_name_to_module(self.model)

        for output_layer_name in self.cfg.hook_layers:
            name_to_module[output_layer_name].register_forward_hook(_getActivation(output_layer_name))

        #for name, module in self.model.named_modules():
        #    #print(name)
        #    if name in self.cfg.hook_layers:
        #        hooks[name.replace('.', '')] = module.register_forward_hook(_getActivation(name))

        return hooks

    def _dummy_batch(self):
        # print("Forwarding a dummy batch")
        x, _ = next(iter(self.loaders['train']))
        y = self.model(x.cuda())
        self.activations['out'] = y
        pool = nn.AdaptiveAvgPool2d(1)
        # if self.cfg.debug:
        #     for k, v in self.activations.items():
        #         print(f"After layer {k}, shape was {v.shape}")

    def check_features_exist(self, mode):
        self._dummy_batch()

        feature_dir = self.feature_dir.parent / mode
        feature_files = list(feature_dir.glob('*.pt'))

        # check which features exist. This line drops the _POSTFIX
        feature_files = ['_'.join(f.stem.split('_')[:-1]) for f in feature_files]
        feature_dir.mkdir(parents=True, exist_ok=True)

        for k in self.activations:
            if f"{k}_{self.dataset_name}" not in feature_files:
                # Something didn't exist
                return False, None

        return True, feature_dir

    def save_features(self, features, targets, mode, postfix):
        feature_dir = self.feature_dir.parent / mode
        feature_dir.mkdir(parents=True, exist_ok=True)

        dict_to_save = {}
        for k in features:
            dict_to_save[k] = {'features':features[k], 'targets':targets[k]}

            feature_file = feature_dir / f"{k}_{self.dataset_name}_{postfix}.pt"
            torch.save(dict_to_save[k], feature_file)
            print(f'saved at {feature_file}')

        dict_to_save = {}


    @torch.no_grad()
    def _extract(self, loader, mode='train', progressbar=True):
        # If already extracted, don't extract
        already_extracted, _fd = self.check_features_exist(mode)
        if already_extracted:
            print(f"All files are alreay extracted! To force feature_extraction, remove all features")
            return

        # dicts to store features and targets for saving on disk
        features={}
        targets={}
        total_processed = 0

        wrapper = ModelWrapper(self.cfg.arch, self.model, self.cfg.hook_layers)

        with tqdm(loader, disable=not progressbar) as t:
            for batch_idx, (inputs, _targets) in enumerate(t): #TODO wrap it with tqdm
                total_processed += inputs.size(0)
                self.activations = {}
                inputs, _targets = inputs.cuda(),_targets.long().squeeze().cuda()
                #out = self.model(inputs)
                out = wrapper(inputs)
                # Get output of the network as a feature
                #self.activations['out'] = out
                #exit()
                self.activations = out

                # Append features and targets to the dictionaries
                for k in self.activations:
                    if k not in features:
                        features[k] = self.activations[k]
                        targets[k] = _targets
                    else:
                        features[k] = torch.cat([features[k], self.activations[k]], dim=0)
                        targets[k] = torch.cat([targets[k], _targets], dim=0)

                # Log progress in TQDM
                t.set_postfix(progress=f"{len(features[k])}/{len(loader.dataset)}")

                # Save features periodically to avoid OOM
                if len(features[k]) + self.cfg.batch_size > self.cfg.features_per_file:
                    self.save_features(features, targets, mode, total_processed)
                    features = {}
                    targets = {}

        if len(features) > 0: # the leftovers
            self.save_features(features, targets, mode, total_processed)
            features = {}
            targets = {}

    def extract(self, progressbar=True):
        modes = ['train', 'val']
        if self.cfg.eval_tst:
            modes.append('test')

        for m in modes:
            print(f"extracting features for {m}")
            loader = self.loaders[m]
            self._extract(loader, m, progressbar)
