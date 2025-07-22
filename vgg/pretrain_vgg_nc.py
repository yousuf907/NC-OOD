import argparse
import os
import random
import shutil
import warnings
import numpy as np
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import time
import timm
import math
import wandb
from tqdm import tqdm
import scipy.linalg as scilin
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler
from vgg_etf import *
from custom_loss import EntropyRegLoss


### This script trains DNN on ImageNet-100 (randomly sampled 100 classes from ImageNet-1K proposed by CMC paper)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--expt_name', type=str)  # name of the experiment
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--ckpt_file', type=str, default='supervised_dnn.pth')
parser.add_argument('--save_dir', type=str, default='supervised_dnn')
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--labels_dir', type=str, default='imagenet_indices')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel') #256
parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay') #1e-4
parser.add_argument('--lr_step_size', default=15, type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr_gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
parser.add_argument('-p', '--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--image_size', type=int, default=224) # resolutions 32,64,128,224
parser.add_argument('--augmentation', action='store_true', help='use augmentation')
parser.add_argument('--alpha', default=0.1, type=float, help='Entropy reg loss weight')
parser.add_argument('--output_dim', type=int, default=0)
parser.add_argument('--hidden_mlp', type=int, default=0)
best_acc1 = 0
best_acc5 = 0

def main():
    args = parser.parse_args()
    print(vars(args))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_acc5
    args.gpu = gpu
    start_time = time.time()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    wandb.init(name=args.expt_name,
    project='Tunnel_Effect_NeXt', entity="yousufovee")
    wandb.config = {
    "learning_rate": args.lr,
    "epochs": args.epochs,
    "batch_size": args.batch_size
    }

    ## Define Architecture
    ## VGG-ETF (output_dim=512, hidden_mlp=2048)
    model = VGG("VGG17", class_num=args.num_classes, output_dim=args.output_dim, hidden_mlp=args.hidden_mlp) # vgg17

    n_parameters = sum(p.numel() for p in model.parameters())
    print('\nNumber of Parameters (in Millions):', n_parameters / 1e6)

    p = count_network_parameters(model)
    print('\nNumber of Trainable Params (in Millions):', p / 1e6)

    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    ### define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()

    ## Entropy Reg Loss
    print("Using EntropyRegLoss with weight ", args.alpha)
    aux_criterion = EntropyRegLoss().cuda()


    # /// Optimizer /// #
    ### AdamW ###
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # /// LR Schedule /// #
    ### Cosine w/ Warmup ###
    warmup_epochs=5
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
             T_max=args.epochs - warmup_epochs, eta_min=0.0)
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_epochs])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    print('\nLoading ImageNet-100 Dataset/Subset') # ImageNet-100
    print("Number of classes:", args.num_classes)
    ## Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    size = int((256 / 224) * args.image_size)

    ### With Augmentations
    if args.augmentation:
        print("Using image augmentations")
        train_dataset = datasets.ImageFolder(traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        print("Not using any image augmentations")
        ### Without Augmentations
        train_dataset = datasets.ImageFolder(traindir,
            transforms.Compose([
            transforms.Resize(size, interpolation=3), # to maintain same ratio w.r.t. 224 images
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(size, interpolation=3), # to maintain same ratio w.r.t. 224 images
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        normalize,
    ]))


    if args.num_classes < 100:
        #################### TRAINING SET #################
        labels = train_dataset.targets
        labels = np.array(labels)  ## necessary
        ## filter out only the indices for the desired class
        train_idx = filter_by_class(labels, min_class=0, max_class=args.num_classes)
        ####################
        print('Number of samples in training dataset', len(train_idx))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=True, sampler=train_sampler, drop_last=False)

        #################### TEST SET ##################
        labels = val_dataset.targets
        labels = np.array(labels)  ## necessary
        ## filter out only the indices for the desired class
        val_idx = filter_by_class(labels, min_class=0, max_class=args.num_classes)
        ####################
        print('Number of samples in test dataset', len(val_idx))
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.workers, pin_memory=True)
        print('Number of samples in training dataset', len(train_dataset))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers, pin_memory=True)
        print('Number of samples in test dataset', len(val_dataset))


    if args.evaluate:
        checkpoint = torch.load(os.path.join(args.save_dir, 'best_' + args.ckpt_file))

        model.load_state_dict(checkpoint['state_dict'])

        acc1_pre, acc5_pre = validate(val_loader, model, criterion, args)
        print("\nBest top1 val accuracy [%]:", acc1_pre.item(), "And best top5 val accuracy [%]:", acc5_pre.item())

        return

    print('\nStarting training...')
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        loss, loss_ce, loss_reg = train_one_epoch(train_loader, model, criterion, aux_criterion, optimizer, epoch, args)

        lr_scheduler.step()

        ## Evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)

        ## Remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)

        ## wandb
        wandb.log({
        "Train_Loss_CE": loss_ce,
        "Train_Loss_Reg": loss_reg,
        "Train_Loss_Total": loss,
        })

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
            }, is_best, args.save_dir, args.ckpt_file)

    ckpt_path = args.save_dir
    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
               f='./' + ckpt_path + '/{}'.format(args.ckpt_file))

    #print("\nBest top1 val accuracy [%]:", best_acc1.item(), "And best top5 val accuracy [%]:", best_acc5.item())
    print('\nBest Top-1 Accuracy: {top1:.2f}\t'
          'Best Top-5 Accuracy: {top5:.2f}'.format(top1=best_acc1.item(), top5=best_acc5.item()))
    spent_time = int((time.time() - start_time) / 60)
    print("\nTotal Runtime (in minutes):", spent_time)


def train_one_epoch(train_loader, model, criterion, aux_criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ## switch to train mode
    model.train()
    start = time.time()
    end = time.time()

    num_iter = len(train_loader)
    loss_arr = np.zeros(num_iter, np.float32)  ## Total Training loss
    ce_loss_arr = np.zeros(num_iter, np.float32) ## CrossEntropy loss
    reg_loss_arr = np.zeros(num_iter, np.float32) ## Entropy Reg loss


    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # print(f"Trained {i} batches in {time.time() - start} secs")

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        ### Compute output
        emb, _, output = model(input, embed=True) # Encoder embeddings, projector embeddings, predictions
        #output = model(input)

        ## CE Loss
        loss1 = criterion(output, target)
        #loss = criterion(output, target)

        ## Entropy Reg Loss
        loss2 = aux_criterion(emb)

        loss = loss1 + args.alpha * loss2

        ## measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_arr[i] = loss.item()
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

        ce_loss_arr[i] = loss1.item()
        reg_loss_arr[i] = loss2.item()

    return np.mean(loss_arr), np.mean(ce_loss_arr), np.mean(reg_loss_arr)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ## switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if 'Auxiliary' in args.arch:
                main_out, aux_out = model(input)
                main_loss = criterion(main_out, target)
                aux_loss = criterion(aux_out, target)
                loss = main_loss + args.auxiliary_weight * aux_loss
                output = main_out
            else:
                output = model(input)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])

def save_checkpoint(state, is_best, save_dir, ckpt_file):
    torch.save(state, os.path.join(save_dir, ckpt_file))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, ckpt_file), os.path.join(save_dir, 'best_' + ckpt_file))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

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
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def filter_by_class(labels, min_class, max_class):
    return list(np.where(np.logical_and(labels >= min_class, labels < max_class))[0])


if __name__ == '__main__':
    main()
