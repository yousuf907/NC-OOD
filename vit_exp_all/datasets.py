# --------------------------------------------------------
# The following code is based on 2 codebases:
## (1) A-ViT
# https://github.com/NVlabs/A-ViT
# Copyright (C) 2022 NVIDIA Corporation. All rights reserved.
## (2) DeiT
# https://github.com/facebookresearch/deit
# Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
# The code is modified to accomodate ViT training
# --------------------------------------------------------

import os
import json
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from ood_datasets import *


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if is_train:
        split = 'trainval'
    else:
        split = 'test'

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train,download=False, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNETR':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 100
    elif args.data_set == 'NINCO':
        root = args.data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 64
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019, category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    elif args.data_set == 'Flower102':
        dataset = Flowers102(args.data_path, split=split, download=False, transform=transform)
        nb_classes = 102

    elif args.data_set == 'CUB200':
        dataset = CUB200(args.data_path, train=is_train, download=False, transform=transform).data
        nb_classes = 100 #200

    elif args.data_set == 'Scene67':
        dataset = Scene67(args.data_path, train=is_train, download=False, transform=transform).data
        nb_classes = 67

    elif args.data_set == 'FGVCAircraft':
        dataset = datasets.FGVCAircraft(args.data_path, split=split, download=False, transform=transform)
        nb_classes = 100

    elif args.data_set == 'MIT67':
        dataset = MIT67(args.data_path, split=split, download=True, transform=transform)
        nb_classes = 67

    elif args.data_set == 'Pet':
        dataset = datasets.OxfordIIITPet(args.data_path, split=split, download=False, transform=transform)
        nb_classes = 37

    elif args.data_set == 'Stl':
        dataset = datasets.STL10(args.data_path, split=split, download=True, transform=transform)
        nb_classes = 10

    return dataset, nb_classes


### With Augmentations
def build_transform(is_train, args):
    #resize_im = args.input_size > 32
    resize_im = args.input_size > 30

    ## train set ##
    if is_train:
        t = []
        t.append(transforms.RandomResizedCrop(args.input_size))
        t.append(transforms.RandomHorizontalFlip())
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

    ## test set ##
    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(transforms.Resize(size, interpolation=3))
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def filter_by_class(labels, min_class, max_class):
    return list(np.where(np.logical_and(labels >= min_class, labels < max_class))[0])


def filter_by_class_balanced(labels, max_class, samples_per_class=100):
    print("creating new indices")
    indices = []
    for class_idx in range(max_class):
        class_indices = np.where(labels == class_idx)[0]
        if samples_per_class is not None:
            class_indices = np.random.choice(class_indices, samples_per_class, replace=False)

        indices.extend(class_indices)

    #save_path = f"{save_dir}/{dataset_type}_{num_classes}_{samples_per_class}.pt"
    #torch.save(indices, save_path)
    return indices
