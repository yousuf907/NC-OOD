from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset, DataLoader


def get_feature_files(feature_dir:str, split:str):
    """ Get layerwise feature files (A list of features for each file)

    Args:
        feature_dir (str): _description_
        split (str): _description_

    Returns:
        dict {str:list}. LayerID is the key and value is a list of feature files for that layer
    """
    feature_dir = Path(feature_dir).parent / split
    feature_files = list(feature_dir.glob('*.pt'))
    # layers = [f.stem.rsplit('_', 1)[0] for f in feature_files]
    
    # single_file_features =  len(layers) == len(unique(layers))
    # if single_file_features:
    #     out = {l:[ff] for l,ff in zip(layers,feature_files)}
    #     return out
    # else:
    out = {}
    for ff in feature_files:
        l = ff.stem.rsplit('_', 1)[0]
        out[l] = [ff] if l not in out.keys() else out[l] + [ff]
    return out
                
            

class Features(Dataset):
    def __init__(
        self,
        feature_dir: str = 'features/Flowers_-0.01_20_l3/gen9', 
        split: str = "train",
        all_in_ram: bool = True
    ):
        super(Features, self).__init__()        
        assert split in ['train', 'val', 'test'], f"split should be train/val/test but given {split}"
        assert all_in_ram == True, "Current implementation assumes we have enough RAM for all layers"
        # Directory in which all features reside
        self.feature_dir = get_feature_files(feature_dir, split)
        # print(self.feature_dir)
        self.layers = sorted(list(self.feature_dir.keys()))
        #print(len(self.layers))
        self.set_layer_called = False
        
    def set_layer(self, idx):
        self.set_layer_called = True
        
        self.current_layer = self.layers[idx]
        feature_files = self.feature_dir[self.current_layer]
        
        self.features = []
        self.targets = []
        
        for file in feature_files:
            # https://github.com/pytorch/pytorch/issues/40403
            torch_data = torch.load(file, map_location=torch.device('cpu'))
            features = torch_data['features']
            targets = torch_data['targets']
            self.features.extend(features)
            self.targets.extend(targets)
        return self.current_layer, features.shape[1]
    
    def len_layers(self):
        print(f"layers are\n {[f'{idx}:{l}' for idx, l in enumerate(self.layers)]}")
        print("Number of layers:", len(self.layers))
        return len(self.layers)
        
        
    def __getitem__(self, idx):
        assert self.set_layer_called, "User must call set_layer before getting items"
        feature = self.features[idx]
        target = self.targets[idx]
        return feature, target
            
    def __len__(self):
        return len(self.features)   


def load_features(cfg, feature_dir):
    trainset = Features(feature_dir, split='train', all_in_ram=cfg.knn_all_in_ram)
    valset = Features(feature_dir, split='val', all_in_ram=cfg.knn_all_in_ram)
    testset = Features(feature_dir, split='test', all_in_ram=cfg.knn_all_in_ram)

    trainloader = DataLoader(trainset, batch_size=cfg.knn_train_batch_size, num_workers=cfg.knn_num_workers, pin_memory=True)
    valloader = DataLoader(valset, batch_size=cfg.knn_test_batch_size, num_workers=cfg.knn_num_workers, pin_memory=True)
    testloader = None
    
    if cfg.eval_tst:
        testloader = DataLoader(testset, batch_size=cfg.knn_test_batch_size, num_workers=cfg.knn_num_workers, pin_memory=True)
        if cfg.knn_test_only:
            valloader = None
    
    return trainloader, valloader, testloader

if __name__=="__main__":
    cfg = SimpleNamespace(knn_all_in_ram=1, knn_num_workers=6, eval_tst=1, knn_test_only=1, knn_train_batch_size=2, knn_test_batch_size=2)
    # dataset = Features(feature_dir='features/Flowers_-0.01_20_l3/gen9', split='train')
    tl, vl, tesl = load_features(cfg, 'features/Flowers_-0.01_20_l3/gen9/best_val.pt')
    tl.dataset.set_layer(2)
    tesl.dataset.set_layer(2)
    print(tesl)