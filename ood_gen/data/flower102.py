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


class Flowers102(datasets.Flowers102):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(Flowers102, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))

        self.targets = []
        self._image_files = []
        for image_id in image_ids:
            self.targets.append(image_id_to_label[image_id] - 1) # -1 for 0-based indexing
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")
        self.classes = list(set(self.targets))
    
    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self.targets[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)


'''
from pathlib import Path
from PIL import Image
from typing import Optional, Callable
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms as T


class Flower102(Dataset):
    def __init__(
        self,
        data_path: str = '../../Datasets/flower102/', # should have jpg and and lists inside. In lists, there should be train/val/test split CSVs
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Optional[int] = 256,
        all_in_ram: bool = False
    ):
        super(Flower102, self).__init__()
        assert split in ['train', 'val', 'test'], f"split should be train/val/test but given {split}"
        
        if split.lower() == 'train':
            csv_file = Path(data_path, 'lists/trn.csv')
        elif split.lower() == 'val':
            csv_file = Path(data_path, 'lists/val.csv')
        elif split.lower() == 'test':
            csv_file = Path(data_path, 'lists/tst.csv')
            
        data_df = pd.read_csv(csv_file)
        images = data_df['file_name'].tolist()
        labels = data_df['label'].tolist()
        cls_name_to_idx = np.sort(np.unique(labels))
        cls_name_to_idx = {k:v for k,v in enumerate(cls_name_to_idx)}
        cls_idx_to_name = {v:k for k,v in cls_name_to_idx.items()}
        
        self.dataset = []
        for idx in range(len(images)):
            img_path = Path(data_path, 'jpg/', images[idx])
            if all_in_ram:
                img_path = Image.open(img_path).convert("RGB")
                
            label = cls_name_to_idx[labels[idx]]
            self.dataset.append((img_path, label))
        
        
        if transform is None:
            transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        
        self.targets = [v for (_, v) in self.dataset]
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
    dataset = Flower102(split='test', all_in_ram=True)
    print(dataset[2][0].shape)
    print(dataset.get_class(5))
    # image, label = dataset[2]
    # print(image.shape)
    # print(label, dataset.cls_idx_to_name[label])

'''