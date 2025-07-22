from pathlib import Path
from PIL import Image
from typing import Optional, Callable
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms as T

class CUB200(Dataset):
    def __init__(
        self,
        data_path: str = '../../Datasets/CUB200/CUB_200_2011/CUB_200_2011/', # should have jpg and and lists inside. In lists, there should be train/val/test split CSVs
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Optional[int] = 256,
        all_in_ram: bool = False
    ):
        super(CUB200, self).__init__()
        assert split in ['train', 'val'], f"split should be train/val but given {split}"
        data_path = Path(data_path)

        labels = pd.read_csv(data_path/"image_class_labels.txt", header=None, sep=" ")
        train_val = pd.read_csv(data_path/'train_test_split.txt', header=None, sep=" ")
        image_names = pd.read_csv(data_path/"images.txt", header=None, sep=" ")
        classes = pd.read_csv(Path(data_path,"classes.txt"), header=None, sep=" ")
        
        labels.columns = ["id", "label"]
        train_val.columns = ["id", "is_train"]
        image_names.columns = ["id", "name"]
        classes.columns = ["id", "class"]
        
        mask = train_val.is_train.values == 1 if split=="train" else train_val.is_train.values == 0
        
        filenames = image_names[mask]["name"].tolist()
        labels = labels[mask]["label"].tolist()
        
        self.dataset = []
        for idx, img_name in enumerate(filenames):
            img_path = data_path/"images"/img_name
            if all_in_ram:
                img_path = Image.open(img_path).convert("RGB")
            label = labels[idx] - 1
            self.dataset.append((img_path, label))
        
        clslist = classes['class'].tolist()
        cls_name_to_idx = {clsname:idx for idx, clsname in enumerate(clslist)}
        cls_idx_to_name = {idx:clsname for clsname, idx in cls_name_to_idx.items()}
        
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
        return img_tensor, label
            
    def __len__(self):
        return len(self.dataset)   

if __name__ == "__main__":
    dataset = CUB200(split='val')
    print(dataset[2][0].shape, dataset[2][1])