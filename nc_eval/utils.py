
from pathlib import Path
import abc
import torch
from enum import Enum, auto

def checkpoint_summary(ckpt_dir: Path, ckpt: dict):
    ckpt_info = '/'.join(ckpt_dir.parts[-3:-1])
    out = f"checkpoint {ckpt_info}\n"
    
    keys = ['gen_best_test_acc', 'best_test_acc', 'cur_val_acc', 'best_val_acc', 'gen_best_val_acc']
    tmp_out = ""
    for k, v in ckpt.items():
        if k in keys:
            tmp_out += k + '=' + str(round(v,4)) + ', '
        if k == 'gen':
            gen = v
        if k == 'epoch_in_gen':
            epoch = v
    
    out = out + f"At [gen {gen}, epoch {epoch}]: " + tmp_out[:-2]
    
    print(out)
    
    
    
class Meter(object):
    @abc.abstractmethod
    def __init__(self, name, fmt=":f"):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def update(self, val, n=1):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass
    
class AverageMeter(Meter):
    """ Computes and stores the average and current value """
    def __init__(self, name, fmt=":f", write_val=True, write_avg=True):
        self.name = name
        self.fmt = fmt
        self.reset()

        self.write_val = write_val
        self.write_avg = write_avg

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def knn_accuracy(pred, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    batch_size = target.size(0)

    res = {}
    for k in topk:
        top_preds, _ = torch.mode(pred[:, :k], dim=-1)
        correct_k = top_preds.eq(target).float().sum()
        res[k] = correct_k.mul_(100.0 / batch_size)
        
    return res
   
    
class Stage(Enum):
    """Simple enum to track stage of experiments."""
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    pass