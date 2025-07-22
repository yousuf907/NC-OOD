import argparse
from types import SimpleNamespace
from pathlib import Path
from typing import Union, Tuple, List

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Layer Probe")
        parser.add_argument('--expt_name', type=str)
        # TASK
        parser.add_argument("--task", default='transfer_linear', type=str, # transfer_linear
                            help='knn_probe (figure 2) or transfer_linear (table 1)',
                            choices=
                            [
                                'knn_probe',
                                'lin_probe',
                                'transfer_linear'
                            ]
                            )
        # Data
        parser.add_argument("--data_dir", default='/', help="path to dataset ROOT directory")
        parser.add_argument(
            "--set",
            type=str,
            default="tinyImagenet_full",
            choices=[
                "tinyImagenet_full",
                "Flower102",
                "CIFAR10",
                "CIFAR100",
                "MNIST",
                "CUB200",
                "MIT67",
                "Scene67",
                "Dog120",
                "Aircrafts",
                "imagenet64",
                "imagenet100",
                "imagenet_r200",
                "ninco",
                "Pet",
                "Stl",
                "imagenet10",
                "imagenet20"
            ],
            help="name of the dataset"
        )
        parser.add_argument("--save_dir", default='/', help="path to save results")
        parser.add_argument("--num_workers", default=6, type=int, help="Number of workers for dataloader")
        parser.add_argument("--all_in_ram", default=0, type=int, help="Load the whole dataset in ram or just the addresses.")
        parser.add_argument("-b", "--batch_size", default=32, type=int, metavar="N", help="mini-batch size.")
        parser.add_argument("--seed", default=0, type=int, help="Seed for randomness")
        # Model
        parser.add_argument("-a", "--arch", metavar="ARCH", default="ResNet18", choices=['ResNet18', 'ResNet34', 'ResNet50', 'vgg11', 'vgg17'])

        # Features
        parser.add_argument("--hook_layers", default=None, type=list, help='Layers on which you wish to linear probe')
        parser.add_argument('--features_root', default="./features/", type=str,help='root directory to save features for KNN')
        parser.add_argument('--features_per_file', default=1000, type=int, help='maximum amount of features in each file (to avoid OOM)')

        # Checkpoints
        parser.add_argument('--ckpt_root', default="./checkpoints", type=str,help='root directory where models checkpoints exist')# used to be ./checkpoints/
        parser.add_argument(
            '--ckpt_paths',
            default = [
                "TnyImagenet_normal/gen1/best_val.pt",
                "TnyImagenet_normal/gen10/best_val.pt",
                "TnyImagenet_LLF/gen10/best_val.pt",
                "TnyImagenet_SEAL/gen10/best_val.pt",
                ],
            type=str,
            help="Please note that the addresses must contain at least 3 parts (including the cfg.ckpt_root it must be 4 parts)! If it doesn't, please create some dummy sub directories to avoid errors"
        )
        parser.add_argument(
            '--ckpt_info',
            default=[
                "Normal-gen1",
                "Normal-gen10",
                "LLF-gen10",
                "SEAL-gen10"
            ],
            type=str,
            help="info for each ckpt's model. (optional)"
        )

        # wandb
        parser.add_argument(
            "--no_wandb", type=int, default=0, help="no wandb"
        )
        parser.add_argument(
            "--wandb_project_name", type=str, default="",
            help="Wandb sweep may overwrite this!",
        )
        parser.add_argument("--wandb_experiment_name", type=str, default="")
        parser.add_argument("--wandb_entity", type=str)
        parser.add_argument("--wandb_offline", type=str, default="online")
        parser.add_argument("--input_size", default=224, type=int, metavar="N", help="input image resolution.")

        # debugging
        parser.add_argument(
            "--debug",
            default=0,
            type = int,
            help="Run the experiment with only a few batches for all"
            "datasets, to ensure code runs without crashing.",
        )
        parser.add_argument(
            "--debug_batches",
            type=int,
            default=2,
            help="Number of batches to run in debug mode.",
        )
        parser.add_argument(
            "--debug_epochs",
            type=int,
            default=3,
            help="Number of epochs to run in debug mode. "
            "If a non-positive number is passed, all epochs are run.",
        )



        parser.add_argument("--extract_features", default=0, type=int, help="Extract features?")

        parser.add_argument("--knn_num_workers", default=6, type=int, help="Number of workers for dataloader")
        parser.add_argument("--knn_all_in_ram", default=1, type=int, help="Load the whole dataset in ram or just the addresses.")
        parser.add_argument("--knn_train_batch_size", default=256, type=int, metavar="N", help="train mini-batch size.")
        parser.add_argument("--knn_test_batch_size", default=128, type=int, metavar="N", help="val/test mini-batch size.")
        parser.add_argument("--knn_test_only", default=1, type=int, metavar="N", help="Do not perform KNN on validation set if test set is available.")
        parser.add_argument("--knn_K", default=[1,], type=Union[List[int], Tuple[int, ...]], help="K for KNN")
        parser.add_argument("--knn_norm", default=1, type=int, help="The order of NORM for KNN")

        # Transfer learning hyper params
        parser.add_argument("--src_ds", default='MiniBestHypers', help="source dataset")

        # Transfer with retraining
        parser.add_argument("--optimizer", default='sgd', type=str, help="Number of workers for dataloader")
        parser.add_argument("--lr", default=1e-2, type=float, help="Number of workers for dataloader")
        parser.add_argument("--momentum", default=0.9, type=float, help="Number of workers for dataloader")
        parser.add_argument("--weight_decay", default=1e-4, type=float, help="Number of workers for dataloader")
        parser.add_argument("--epochs", default=30, type=int, help="Number of epochs for training")
        parser.add_argument("--nc_control", type=int, default=1, help="control nc") # nc control?, yes>1, no>0

        self.parser = parser

    def parse(self, args):
        self.cfg = self.parser.parse_args(args)
        Defaults = {
            "tinyImagenet_full": {"num_cls":200, "eval_tst":False},
            "Flower102": {"num_cls":102, "eval_tst":False},
            "Dog120": {"num_cls":120, "eval_tst":False},
            "CUB200": {"num_cls":200, "eval_tst":False},

            "MIT67": {"num_cls":67, "eval_tst":False},
            "Scene67": {"num_cls":67, "eval_tst":False},
            "Pet": {"num_cls":37, "eval_tst":False},
            "Stl": {"num_cls":10, "eval_tst":False},
            #"Aircrafts": {"num_cls":100, "eval_tst":True},
            "Aircrafts": {"num_cls":100, "eval_tst":False},
        }

        if type(self.cfg.ckpt_paths) == str:
            self.cfg.ckpt_paths = [self.cfg.ckpt_paths]
        if type(self.cfg.ckpt_info) == str:
            self.cfg.ckpt_info = [self.cfg.ckpt_info]

        self.cfg.data_dir = Path(self.cfg.data_dir)
        if self.cfg.set in Defaults:
            setting = Defaults[self.cfg.set]
            self.cfg.num_cls = setting["num_cls"]
            self.cfg.eval_tst = setting["eval_tst"]

        elif (
            self.cfg.set == "CIFAR10"
            or self.cfg.set == "CIFAR10val"
            or "mnist" in self.cfg.set.lower()
        ):
            self.cfg.data_dir = "../Datasets"
            self.cfg.num_cls = 10
            self.cfg.eval_tst = False
        elif self.cfg.set == "CIFAR100" or self.cfg.set == "CIFAR100val":
            #self.cfg.data_dir = "../Datasets"
            self.cfg.num_cls = 100
            self.cfg.eval_tst = False
        #elif self.cfg.set == "imagenet100" or self.cfg.set == "imagenet_r100":
        elif self.cfg.set == "imagenet100":
            self.cfg.num_cls = 100
            self.cfg.eval_tst = False
        elif self.cfg.set == "imagenet10":
            self.cfg.num_cls = 10
            self.cfg.eval_tst = False
        elif self.cfg.set == "imagenet20":
            self.cfg.num_cls = 20
            self.cfg.eval_tst = False
        elif self.cfg.set == "imagenet_r200":
            self.cfg.num_cls = 200
            self.cfg.eval_tst = False
        elif self.cfg.set == "ninco":
            self.cfg.num_cls = 64
            self.cfg.eval_tst = False
        elif self.cfg.set == "CUB200":
            self.cfg.num_cls = 200
            self.cfg.eval_tst = False
        else:
            raise NotImplementedError("Invalid dataset {}".format(self.cfg.set))

        if self.cfg.hook_layers is None:
            if self.cfg.arch.lower() == 'resnet50':
                self.cfg.hook_layers = [
                        "layer2.0.conv1",
                        "layer2.2.conv1",
                        "layer3.0.conv1",
                        "layer3.3.conv1",
                        "layer4.0.conv1",
                        "layer4.2.conv1"
                    ]
            elif self.cfg.arch.lower() == 'resnet18':

                '''
                self.cfg.hook_layers = [
                        "conv1",
                        "layer1.0.conv1",
                        "layer1.0.conv2",
                        "layer1.1.conv1",
                        "layer1.1.conv2",
                        "layer2.0.conv1",
                        "layer2.0.conv2",
                        "layer2.1.conv1",
                        "layer2.1.conv2",
                        "layer3.0.conv1",
                        "layer3.0.conv2",
                        "layer3.1.conv1",
                        "layer3.1.conv2",
                        "layer4.0.conv1", # 14
                        "layer4.0.conv2", # 15
                        "layer4.1.conv1", # 16
                        "layer4.1.conv2",
                        "projection_head.0",
                        "projection_head.2"
                    ]
                '''

                ## ResNet18
                if self.cfg.nc_control:
                
                    self.cfg.hook_layers = [
                            "layer4.1.conv2",
                            "projection_head.2"
                        ]
                else:

                    self.cfg.hook_layers = [
                            "layer4.1.conv2"
                        ]

            elif self.cfg.arch.lower() == 'resnet34':
                '''
                self.cfg.hook_layers = [
                        "conv1",
                        "layer1.0.conv1",
                        "layer1.0.conv2",
                        "layer1.1.conv1",
                        "layer1.1.conv2",
                        "layer1.2.conv1",
                        "layer1.2.conv2",
                        "layer2.0.conv1",
                        "layer2.0.conv2",
                        "layer2.1.conv1",
                        "layer2.1.conv2",
                        "layer2.2.conv1",
                        "layer2.2.conv2",
                        "layer2.3.conv1",
                        "layer2.3.conv2",
                        "layer3.0.conv1",
                        "layer3.0.conv2",
                        "layer3.1.conv1",
                        "layer3.1.conv2",
                        "layer3.2.conv1",
                        "layer3.2.conv2",
                        "layer3.3.conv1",
                        "layer3.3.conv2",
                        "layer3.4.conv1",
                        "layer3.4.conv2",
                        "layer3.5.conv1",
                        "layer3.5.conv2",
                        "layer4.0.conv1",
                        "layer4.0.conv2",
                        "layer4.1.conv1",
                        "layer4.1.conv2",
                        "layer4.2.conv1",
                        "layer4.2.conv2",
                        "projection_head.0",
                        "projection_head.2"
                    ]
                '''
                ## ResNet34
                if self.cfg.nc_control:

                    self.cfg.hook_layers = [
                            "layer4.2.conv2",
                            "projection_head.2"
                        ]
                else:
                    self.cfg.hook_layers = [
                            "layer4.2.conv2"
                        ]


            elif self.cfg.arch.lower() == 'vgg11':
                ## WITH maxpooling for all resolutions
                self.cfg.hook_layers = [
                        "features.0",
                        "features.3",
                        "features.7",
                        "features.10",
                        "features.14",
                        "features.17",
                        "features.21",
                        "features.24",
                        "features.28",
                        "features.31"
                    ]


            elif self.cfg.arch.lower() == 'vgg17':
                '''
                ## VGG17 WITH maxpooling for all resolutions
                self.cfg.hook_layers = [
                        "features.0", #1
                        "features.3", #2
                        "features.7", #3
                        "features.10", #4
                        "features.14", #5
                        "features.17", #6
                        "features.20", #7
                        "features.23", #8
                        "features.27", #9
                        "features.30", #10
                        "features.33", #11
                        "features.36", #12
                        "features.40", #13
                        "features.43", #14
                        "features.46", #15
                        "features.49", #16 <last encoder_layer>
                        "projection_head.0", #17
                        "projection_head.2" #18 <last projector_layer>
                    ]
                '''

                ## VGG17
                if self.cfg.nc_control:

                    self.cfg.hook_layers = [
                            "features.49",
                            "projection_head.2"
                        ]
                else:
                    self.cfg.hook_layers = [
                            "features.49"
                        ]


        if self.cfg.debug:
            self.cfg.val_interval = 2

        if isinstance(self.cfg.ckpt_paths, str):
            self.cfg.ckpt_paths = list(self.cfg.ckpt_paths)

        if len(self.cfg.ckpt_info) != len(self.cfg.ckpt_paths):
            print('bad info! Length should match the number of checkpoints')
            self.cfg.ckpt_info = [f'run{i:2d}' for i in range(len(self.cfg.ckpt_paths))]

        self.cfg.ckpt_full_paths = [
            Path(self.cfg.ckpt_root, x) for x in self.cfg.ckpt_paths
        ]
        ## Create directories to save features
        self.cfg.features_full_paths = [
            Path(self.cfg.features_root, conf) for conf in self.cfg.ckpt_paths
        ]
        for features_fp in self.cfg.features_full_paths:
            features_fp.parent.mkdir(parents=True, exist_ok=True)

        return self.cfg




if __name__ == "__main__":
    pass
