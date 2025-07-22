from pathlib import Path
from PIL import Image
from typing import Optional, Callable
from  scipy.io import loadmat

from torch.utils.data import Dataset
from torchvision import transforms as T

class Dog120(Dataset):
    def __init__(
        self,
        data_path: str = '../../Datasets/Dogs/', # should have jpg and and lists inside. In lists, there should be train/val/test split CSVs
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: Optional[int] = 256,
        all_in_ram: bool = False
    ):
        super(Dog120, self).__init__()
        assert split in ['train', 'val'], f"split should be train/val but given {split}"
        data_path = Path(data_path)

        anno = "train_list.mat" if split == 'train' else "test_list.mat"
        
        paths = loadmat(data_path/anno)['file_list']
        labels = loadmat(data_path/anno)['labels']
        filenames = [path[0][0] for path in paths]
        labels = [label[0]-1 for label in labels]
        
        
        self.dataset = []
        cls_name_to_idx = {}
        for img_name, label in zip(filenames, labels):
            img_path = data_path/"Images"/img_name
            if all_in_ram:
                img_path = Image.open(img_path).convert("RGB")
                
            cls_name = img_name.split('/')[0]
            cls_name_to_idx[cls_name] = label
            self.dataset.append((img_path, label))
        
        cls_idx_to_name = {idx:clsname for clsname, idx in cls_name_to_idx.items()}
        
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

if __name__ == "__main__":
    dataset = Dog120(split='train', all_in_ram=True)
    print(dataset[3000][0].shape, dataset[3000][1])
    # print(dataset.cls_name_to_idx)