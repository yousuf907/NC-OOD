from pathlib import Path
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision import transforms as T

class Aircrafts(Dataset):
    def __init__(
        self,
        data_path: str = '../Datasets/fgvc-aircraft-2013b/', # should have jpg and and lists inside. In lists, there should be train/val/test split CSVs
        split: str = "train",
        annotation_level: str = 'variant',
        transform: Optional[Callable] = None,
        image_size: Optional[int] = 256,
        all_in_ram: bool = True
    ):
        super(Aircrafts, self).__init__()        
        assert split in ['train', 'val', 'trainval', 'test'], f"split should be train/val/test but given {split}"
        
        _annotations = {
                "variant": "variants.txt",
                "family": "families.txt",
                "manufacturer": "manufacturers.txt",
            }
        
        data_path = Path(data_path) / "data"
        anno = _annotations[annotation_level]
        
        with open(data_path / anno, "r") as f:
            self.classes = [line.strip() for line in f]
        
        cls_name_to_idx = dict(zip(self.classes, range(len(self.classes))))
        cls_idx_to_name = {idx:clsname for clsname, idx in cls_name_to_idx.items()}
        
        images_folder = data_path / "images"
        labels_file = data_path / f"images_{annotation_level}_{split}.txt"
        
        self.dataset = []
        
        with open(labels_file, "r") as f:
            for line in f:
                img_path, cls_name = line.strip().split(" ", 1)
                img_path = images_folder / f"{img_path}.jpg"
                if all_in_ram:
                    img_path = Image.open(img_path).convert("RGB")
                
                self.dataset.append((img_path, cls_name_to_idx[cls_name]))
        
        if transform is None:
            transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        
        self.targets = [v for _,v in self.dataset]
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

if __name__=="__main__":
    dataset = Aircrafts()
    img, label = dataset[2]
    print(img.shape, label)