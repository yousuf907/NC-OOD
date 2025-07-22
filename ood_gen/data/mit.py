import os

import os.path
import pathlib
from pathlib import Path

from typing import Any, Tuple, Optional, Callable

import glob
from shutil import move, rmtree

import numpy as np

import torch
from torchvision import datasets
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg, download_and_extract_archive

import PIL
from PIL import Image


import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms as T


class Scene67(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        image_url = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
        train_annos_url = 'http://web.mit.edu/torralba/www/TrainImages.txt'
        test_annos_url = 'http://web.mit.edu/torralba/www/TestImages.txt'
        urls = [image_url, train_annos_url, test_annos_url]
        image_fname = 'indoorCVPR_09.tar'
        self.train_annos_fname = 'TrainImage.txt'
        self.test_annos_fname = 'TestImage.txt'
        fnames = [image_fname, self.train_annos_fname, self.test_annos_fname]

        for url, fname in zip(urls, fnames):
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from ' + url)
                    download_url(url, root, filename=fname)
        if not os.path.exists(os.path.join(root, 'Scene67')):
            import tarfile
            with tarfile.open(os.path.join(root, image_fname)) as tar:
                tar.extractall(os.path.join(root, 'Scene67'))

            self.split()

        if self.train:
            fpath = os.path.join(root, 'Scene67', 'train')

        else:
            fpath = os.path.join(root, 'Scene67', 'test')

        self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        if not os.path.exists(os.path.join(self.root, 'Scene67', 'train')):
            os.mkdir(os.path.join(self.root, 'Scene67', 'train'))
        if not os.path.exists(os.path.join(self.root, 'Scene67', 'test')):
            os.mkdir(os.path.join(self.root, 'Scene67', 'test'))
        
        train_annos_file = os.path.join(self.root, self.train_annos_fname)
        test_annos_file = os.path.join(self.root, self.test_annos_fname)

        with open(train_annos_file, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                src = self.root + 'Scene67/' + 'Images/' + line
                dst = self.root + 'Scene67/' + 'train/' + line
                if not os.path.exists(os.path.join(self.root, 'Scene67', 'train', line.split('/')[0])):
                   os.mkdir(os.path.join(self.root, 'Scene67', 'train', line.split('/')[0]))
                move(src, dst)
        
        with open(test_annos_file, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                src = self.root + 'Scene67/' + 'Images/' + line
                dst = self.root + 'Scene67/' + 'test/' + line
                if not os.path.exists(os.path.join(self.root, 'Scene67', 'test', line.split('/')[0])):
                   os.mkdir(os.path.join(self.root, 'Scene67', 'test', line.split('/')[0]))
                move(src, dst)



class MIT67(Dataset):
    def __init__(
        self,
        data_path: str = '../../Datasets/MIT/', # should have jpg and and lists inside. In lists, there should be train/val/test split CSVs
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Optional[int] = 256,
        all_in_ram: bool = False
    ):
        super(MIT67, self).__init__()
        assert split in ['train', 'val'], f"split should be train/val but given {split}"
        data_path = Path(data_path)

        anno = "TrainImages.txt" if split == 'train' else "TestImages.txt"
        paths = pd.read_csv(data_path/anno, header=None)
        
        paths.columns = ["paths"]
        
        filenames = paths["paths"].tolist()
        
        self.dataset = []
        targets = []
        for idx, img_name in enumerate(filenames):
            img_path = data_path/"Images"/img_name
            if all_in_ram:
                img_path = Image.open(img_path).convert("RGB")
            label = img_name.split('/')[0]
            targets.append(label)
            self.dataset.append((img_path, label))
        
        cls_name_to_idx = {clsname:idx for idx, clsname in enumerate(np.sort(np.unique(targets)))}
        cls_idx_to_name = {idx:clsname for clsname, idx in cls_name_to_idx.items()}
        
        if transform is None:
            transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        
        self.targets = [cls_name_to_idx[target] for target in targets]
        self.all_in_ram = all_in_ram
        self.cls_name_to_idx = cls_name_to_idx
        self.cls_idx_to_name = cls_idx_to_name
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if not self.all_in_ram:
            img = Image.open(img).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, self.cls_name_to_idx[label]
            
    def __len__(self):
        return len(self.dataset)   

if __name__ == "__main__":
    dataset = MIT67(split='train', all_in_ram=True)
    print(dataset[3][0].shape, dataset[3][1])
    # print(dataset.cls_name_to_idx)