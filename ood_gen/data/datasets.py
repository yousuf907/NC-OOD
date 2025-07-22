
import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms as T
import torchvision.datasets as Datasets
import torchvision
from collections import defaultdict
import os
import random
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from torchvision.datasets import ImageFolder
import numpy as np
from .tinyImageNet import TinyImageNet
from .flower102 import Flowers102
from .cub import CUB200
from .mit import MIT67, Scene67
from .dogs import Dog120
from .aircrafts import Aircrafts
from pathlib import Path

class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices
#             yield list(itertools.chain(*zip(batch_indices,pair_indices )))

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations

def filter_by_class(labels, min_class, max_class):
    return list(np.where(np.logical_and(labels >= min_class, labels < max_class))[0])

def _get_mean_std(cfg):
    if cfg.set.lower() == 'cifar10':
        mean_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615)
    elif cfg.set.lower() == 'cifar100':
        mean_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    elif 'mnist' in cfg.set.lower():
        mean_std = (0.1307), (0.3081)
    elif 'imagenet64':
        mean_std=(0.482, 0.458, 0.408), (0.269, 0.261, 0.276)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return mean_std

def get_transforms(cfg):
    if 'mnist' in cfg.set.lower():
        transform_train = T.Compose([
            T.ToTensor(),
        ])
        transform_test = T.Compose([
            T.ToTensor(),
        ])

    ## ImageNet-100 subset
    elif cfg.set in ['imagenet100', 'imagenet_r200']:
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        size = int((256 / 224) * cfg.input_size)
        transform_train = T.Compose([
            T.Resize(size, interpolation=3), # to maintain same ratio w.r.t. 224 images
            T.CenterCrop(cfg.input_size),
            T.ToTensor(),
            normalize,
        ])
        transform_test = T.Compose([
            T.Resize(size, interpolation=3), # to maintain same ratio w.r.t. 224 images
            T.CenterCrop(cfg.input_size),
            T.ToTensor(),
            normalize,
        ])


    elif cfg.set in ['CIFAR10', 'CIFAR100']: ## CIFAR10 and CIFAR10 as ID datasets
        if cfg.input_size == 32:
            transform_train = T.Compose([
            T.ToTensor(),
            T.Normalize(*_get_mean_std(cfg))
            ])
            transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize(*_get_mean_std(cfg))
            ])
        else:
            size = int((256 / 224) * cfg.input_size)
            transform_train = T.Compose([
                T.Resize(size, interpolation=3),
                T.CenterCrop(cfg.input_size),
                T.ToTensor(),
                T.Normalize(*_get_mean_std(cfg))
                ])
            transform_test = T.Compose([
                T.Resize(size, interpolation=3),
                T.CenterCrop(cfg.input_size),
                T.ToTensor(),
                T.Normalize(*_get_mean_std(cfg))
                ])

    elif cfg.set in ['tinyImagenet_full', 'imagenet64']: # 64x64
        transform_train = T.Compose([
            T.ToTensor(),
            T.Normalize(*_get_mean_std(cfg))
        ])
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize(*_get_mean_std(cfg))
        ])

    elif cfg.set in ['CUB200', 'STANFORD120', 'MIT67', 'Scene67', 'Aircrafts', 'Pet', 'Stl', 'Dog120', 'Flower102','CUB200_val', 'Dog120_val', 'MIT67_val']:
        size = int((256 / 224) * cfg.input_size)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = T.Compose([
            T.Resize(size, interpolation=3),
            T.CenterCrop(cfg.input_size),
            T.ToTensor(),
            normalize
            ])
        transform_test = T.Compose([
            T.Resize(size, interpolation=3),
            T.CenterCrop(cfg.input_size),
            T.ToTensor(),
            normalize
            ])
    else:
        return None, None
    return transform_train, transform_test


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, torchvision.datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]


def load_dataset(cfg):
    transform_train, transform_test = get_transforms(cfg)
    testset = None
    testloader = None
    if cfg.set in ['CIFAR10', 'CIFAR10_10p']:
        trainset = Datasets.CIFAR10(
                    root=cfg.data_dir, train=True, download=True, transform=transform_train)
        valset = Datasets.CIFAR10(
                    root=cfg.data_dir, train=False, download=True, transform=transform_test)

    elif cfg.set=='CIFAR100':
        trainset = Datasets.CIFAR100(
            root=cfg.data_dir, train=True, download=True, transform=transform_train)
        valset = Datasets.CIFAR100(
            root=cfg.data_dir, train=False, download=True, transform=transform_test)

    elif cfg.set=='MNIST':
        trainset = Datasets.MNIST(
            root=cfg.data_dir, train=True, download=True, transform=transform_train)
        valset = Datasets.MNIST(
            root=cfg.data_dir, train=False, download=True, transform=transform_test)

    elif cfg.set=='tinyImagenet_full':
        trainset = TinyImageNet(cfg.data_dir, split='train', transform=transform_train, all_in_ram=cfg.all_in_ram)
        valset = TinyImageNet(cfg.data_dir, split='val', transform=transform_test, all_in_ram=cfg.all_in_ram)

    elif cfg.set=='imagenet64':
        trainset = ImageFolder(Path(cfg.data_dir)/'train', transform=transform_train)
        valset = ImageFolder(Path(cfg.data_dir)/'val', transform=transform_test)

    elif cfg.set=='Flower102':
        trainset = Flowers102(cfg.data_dir, split='train', download=True, transform=transform_train)
        valset = Flowers102(cfg.data_dir, split='val', download=True, transform=transform_test)
        #trainset = Flowers102(cfg.data_dir, split='test', download=False, transform=transform_test)

    elif cfg.set=='Pet':
        trainset = Datasets.OxfordIIITPet(cfg.data_dir, split='trainval', download=True, transform=transform_train)
        valset = Datasets.OxfordIIITPet(cfg.data_dir, split='test', download=True, transform=transform_test)

    elif cfg.set=='MIT67':
        trainset = MIT67(cfg.data_dir, split='train', transform=transform_train, all_in_ram=cfg.all_in_ram)
        valset = MIT67(cfg.data_dir, split='val', transform=transform_test, all_in_ram=cfg.all_in_ram)

    elif cfg.set=='Scene67':
        trainset = Scene67(cfg.data_dir, train=True, download=True, transform=transform_train).data
        trainset = Scene67(cfg.data_dir, train=False, download=True, transform=transform_test).data

    elif cfg.set=='Dog120':
        trainset = Dog120(cfg.data_dir, split='train', transform=transform_train, all_in_ram=cfg.all_in_ram)
        valset = Dog120(cfg.data_dir, split='val', transform=transform_test, all_in_ram=cfg.all_in_ram)

    elif cfg.set=='CUB200':
        cfg.data_dir = '/home/yousuf/cub200'
        traindir = os.path.join(cfg.data_dir, 'train')
        valdir = os.path.join(cfg.data_dir, 'val')
        trainset = Datasets.ImageFolder(traindir, transform_train)
        valset = Datasets.ImageFolder(valdir, transform_test)

    elif cfg.set == 'Aircrafts':
        cfg.data_dir='./data/fgvc-aircraft-2013b'
        trainset = Aircrafts(cfg.data_dir, split='trainval', transform = transform_train, all_in_ram=cfg.all_in_ram)
        valset = Aircrafts(cfg.data_dir, split='test', transform = transform_test, all_in_ram=cfg.all_in_ram)
        #testset = Aircrafts(cfg.data_dir, split='test', transform = transform_test, all_in_ram=cfg.all_in_ram)

    elif cfg.set == 'Stl':
        trainset = Datasets.STL10(cfg.data_dir, split='train', download=True, transform=transform_train)
        valset = Datasets.STL10(cfg.data_dir, split='test', download=True, transform=transform_test)

    ## Imagenet-100 subset
    elif cfg.set=='imagenet100':
        cfg.data_dir='/data/datasets/ImageNet-100'
        traindir = os.path.join(cfg.data_dir, 'train')
        trainset = Datasets.ImageFolder(traindir, transform_train)
        valdir = os.path.join(cfg.data_dir, 'val')
        valset = Datasets.ImageFolder(valdir, transform_test)

    elif cfg.set=='imagenet_r200':
        cfg.data_dir='./data/imagenet-r'
        traindir = os.path.join(cfg.data_dir, 'train')
        trainset = Datasets.ImageFolder(traindir, transform_train)
        valdir = os.path.join(cfg.data_dir, 'test')
        valset = Datasets.ImageFolder(valdir, transform_test)

    if cfg.set in ['Aircrafts', 'CUB200', 'Dog120', 'MIT67', 'Scene67', 'Flower102', 'imagenet64', 'tinyImagenet_full']:
        trainset = DatasetWrapper(trainset)
        valset = DatasetWrapper(valset)
        if testset is not None:
            testset = DatasetWrapper(testset)

    # TODO: If knn, Sequential
    if cfg.task == 'feature_extraction':
        get_train_sampler = lambda d: BatchSampler(SequentialSampler(d), cfg.batch_size, False)
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), cfg.batch_size, False)
    else:
        get_train_sampler = lambda d: BatchSampler(RandomSampler(d), cfg.batch_size, False)
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), cfg.batch_size, False)

    trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=cfg.num_workers, pin_memory=True)
    valloader = DataLoader(valset,   batch_sampler=get_test_sampler(valset), num_workers=cfg.num_workers, pin_memory=True)

    if testset is not None:
        testloader = DataLoader(testset,   batch_sampler=get_test_sampler(testset), num_workers=cfg.num_workers, pin_memory=True)

    return trainloader, valloader, testloader



if __name__ == '__main__':
    from config import Config
    config = Config().parse(None)
    # config.set = 'CIFAR10'
    config.batch_size = 32
    print(config)
    print(config.set)
    print(config.data_dir)
    tl, vl, _ = load_dataset(config)
    # print(len(vl)*config.batch_size)
    print(len(tl))
    batch = next(iter(tl))

    # print(batch[0].shape, batch[1].shape)
    # print(batch[1])
