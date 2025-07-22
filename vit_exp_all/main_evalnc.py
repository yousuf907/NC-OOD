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
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
import torch.nn as nn
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from timm.models.layers import trunc_normal_
from datasets_lp import build_dataset, filter_by_class
from engine_evalnc import train_one_epoch, evaluate, ood_evaluate, evaluate_ckpt, visualize, nc_values, nc_analysis
from losses import DistillationLoss
from samplers import RASampler
import models_act
import utils
import os
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler


def get_args_parser():
    parser = argparse.ArgumentParser('ViT linear probe training and evaluation script', add_help=False)
    parser.add_argument('--expt_name', type=str)  # name of the experiment
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['IMNETR','NINCO','CIFAR','IMNET','INAT','INAT19',
                        'Flower102','CUB200','Scene67','Aircraft','MIT67','Pet','Stl'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    ## distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # parameters for ViT
    parser.add_argument('--ponder_token_scale', default=0.001, type=float, help="")
    parser.add_argument('--pretrained', action='store_true',
                        help='raise to load pretrained.')
    parser.add_argument('--act_mode', default=4, type=int,
                        help='4-token act, make sure this is always 4, other modes are only used for initial method comparison and exploration')
    parser.add_argument('--tensorboard', action='store_true',
                        help='raise to load pretrained.')
    parser.add_argument('--gate_scale', default=100., type=float, help="constant for token control gate rescale")
    parser.add_argument('--gate_center', default= 3., type=float, help="constant for token control gate re-center, negatived when applied")
    parser.add_argument('--warmup_epoch', default=0, type=int, help="warm up epochs for act")
    parser.add_argument('--distr_prior_alpha', default=0.01, type=float, help="scaling for kl of distributional prior")

    ## a sample visualiztion of tiny for token depth and attention intepretation.
    parser.add_argument('--demo', action='store_true', help='raise to visualize a demo token depth distribution.')
    parser.add_argument('--num_layers', type=int, default=12) # num blocks in ViT
    return parser


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    print("ID Dataset :", args.data_set)

    ## ImageNet (ID)
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    print("Number of classes in the dataset (ImageNet100):", args.nb_classes)

    if True:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print('\nSize of train loader (ImageNet100)', len(dataset_train))
    print('\nSize of val loader (ImageNet100)', len(dataset_val))


    ## NINCO (OOD)
    args.data_set = 'NINCO'
    print("OOD Dataset :", args.data_set)
    dset, ood_classes = build_dataset(is_train=False, args=args)
    print("\nNumber of classes (Ninco):", ood_classes)
    train_idx, valid_idx = train_test_split(
        np.arange(len(dset)), test_size=0.2, random_state=args.seed, stratify=dset.targets)
    train_dataset = Subset(dset, train_idx)
    valid_dataset = Subset(dset, valid_idx)
    print(f"size of train dataset (Ninco): {len(train_dataset)}")
    print(f"size of test dataset (Ninco): {len(valid_dataset)}")
    data_loader_train2 = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val2 = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        pretrained=False, #args.pretrained,
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        args=args
    )

    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    n_parameters = sum(p.numel() for p in model.parameters())
    print('\nNumber of Params:', n_parameters)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)

    ## Freeze model
    for _, p in model.named_parameters():
        p.requires_grad = False

    model.to(device)

    ### For NC Eval
    if args.model == 'vit_tiny':
        args.num_layers=1
        embed_dim=192
    elif args.model == 'vit_small':
        args.num_layers=1
        embed_dim=384
    elif args.model == 'evit_tiny':
        args.num_layers=2 # +1 for projector layer
        embed_dim=192
    elif args.model == 'evit_small':
        args.num_layers=2 # +1 for projector layer
        embed_dim=384
    print("\nCheck embedding dim:", embed_dim)
    print("\nNumber of ViT Layers to Process:", args.num_layers)


    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    #if args.distributed:
    #    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
    #    model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = []
    linear_classifier = []
    for i in range(args.num_layers):
        each_classifier = LinearClassifier(
        dim=embed_dim,
        #dim=embed_dim * (1 + int(avgpool_patchtokens)),
        num_labels=args.nb_classes)
        each_classifier = each_classifier.cuda()
        if utils.get_world_size() > 1:
            each_classifier = nn.parallel.DistributedDataParallel(each_classifier, device_ids=[args.gpu])

        linear_classifier.append(each_classifier)

        optimizer.append(torch.optim.AdamW(
            each_classifier.parameters(), args.lr, weight_decay=1e-4,
        ))

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        print("Using label smoothing")
        #criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            #if 'scaler' in checkpoint:
            #    loss_scaler.load_state_dict(checkpoint['scaler'])

    tf_writer=None
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        tf_path = os.path.join(Path.cwd(),args.output_dir)
        try:
            tf_writer = SummaryWriter(log_dir="%s" % (tf_path))
        except:
            tf_writer = SummaryWriter(logdir="%s" % (tf_path))

    if args.demo:
        visualize(data_loader_val, model, device, tf_writer=tf_writer, epoch=0, args=args)
        # This function can embed in other analysis too. This is a quick example code only for the visualize function.
        return

    if args.eval:
        #test_stats = evaluate(data_loader_val, model, device, tf_writer=tf_writer, epoch=0, args=args)
        test_stats = evaluate_ckpt(data_loader_val, model, device, epoch=0, args=args)
        #print(f"Accuracy of the network on the {len(val_idx)} test images: {test_stats['acc1']:.2f}%")
        print(f"Accuracy of the network on the test images: {test_stats['acc1']:.2f}%")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_acc = 0.0
    best_test_stats = torch.zeros((args.num_layers), dtype=torch.float32)
    lp_acc = []
    best_preds = torch.zeros((args.num_layers, len(data_loader_val.dataset)), dtype=torch.float32)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            if not args.data_set == 'NINCO':
                data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
                model, linear_classifier, criterion, data_loader_train,
                optimizer, device, epoch,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=args.finetune == '',
                args=args, tf_writer=tf_writer  # keep in eval mode during finetuning
            )

        test_stats, preds, all_lbls = evaluate(data_loader_val, model, linear_classifier, device, epoch, tf_writer=tf_writer, args=args)

        l=0
        for classifier_key in test_stats:
            classifier = test_stats[classifier_key]
            #l += 1
            print(f"Top1 Accuracy at epoch {epoch} of layer {l+1} on the test images: {classifier['acc1']:.2f}%")
            best_acc = max(best_acc, classifier["acc1"])

            is_best = classifier["acc1"] > best_test_stats[l]
            if epoch > 0:
                best_test_stats[l] = max(best_test_stats[l], classifier["acc1"])
            else:
                best_preds[l, :] = preds[l, :]
                best_test_stats[l] = classifier["acc1"]

            if (epoch+1) == args.epochs:
                best_acc1 = best_test_stats[l]
                lp_acc = np.append(lp_acc, best_acc1.cpu().numpy())

            if is_best:
                best_preds[l, :] = preds[l, :]

            ## increment layer number
            l += 1

        print("\nBest Top1 accuracy of all layers:", best_test_stats)
        print(f'Max accuracy among layers: {best_acc:.2f}%') # to globally find optimum epoch for best accuracy

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")


    ## ///////////// NC eval ///////////////

    nc1=[]
    nc2=[]
    nc3=[]
    nc4=[]
    sw=[]
    sb=[]
    t=0
    for lc in linear_classifier:
        t+=1
        if t == 1: # encoder
            ## Embeddings from projector (True)/ encoder (False)
            W_ls, mu_G, mu_c_dict,  Sigma_W, Sigma_B = nc_analysis(model, args.nb_classes, data_loader_train, args, proj=False)
            ##
            Wo = lc.state_dict()['module.linear.weight']
            bo = lc.state_dict()['module.linear.bias']

            ## NC metrics
            nc1_e, nc2_e, nc3_e, nc4_e = nc_values(Sigma_W.cpu().numpy(), Sigma_B.cpu().numpy(), mu_c_dict, mu_G, Wo, bo)

            sw_trace = np.trace(Sigma_W.cpu().numpy())
            sb_trace = np.trace(Sigma_B.cpu().numpy())

            nc1 = np.append(nc1, nc1_e)
            nc2 = np.append(nc2, nc2_e)
            nc3 = np.append(nc3, nc3_e)
            nc4 = np.append(nc4, nc4_e)

            sw = np.append(sw, sw_trace)
            sb = np.append(sb, sb_trace)

        elif t == 2:

            ## Embeddings from projector (True)
            W_ls_p, mu_G_p, mu_c_dict_p,  Sigma_W_p, Sigma_B_p = nc_analysis(model, args.nb_classes, data_loader_train, args, proj=True)
            ##
            Wo_p = lc.state_dict()['module.linear.weight']
            bo_p = lc.state_dict()['module.linear.bias']

            ## NC metrics
            nc1_p, nc2_p, nc3_p, nc4_p = nc_values(Sigma_W_p.cpu().numpy(), Sigma_B_p.cpu().numpy(), mu_c_dict_p, mu_G_p, Wo_p, bo_p)

            sw_trace_p = np.trace(Sigma_W_p.cpu().numpy())
            sb_trace_p = np.trace(Sigma_B_p.cpu().numpy())

            nc1 = np.append(nc1, nc1_p)
            nc2 = np.append(nc2, nc2_p)
            nc3 = np.append(nc3, nc3_p)
            nc4 = np.append(nc4, nc4_p)
            sw = np.append(sw, sw_trace_p)
            sb = np.append(sb, sb_trace_p)

    ## Uncomment for saving results
    '''
    filename4 = args.expt_name + '_nc1.npy'
    filename5 = args.expt_name + '_nc2.npy'
    filename6 = args.expt_name + '_nc3.npy'
    filename7 = args.expt_name + '_nc4.npy'
    filename8 = args.expt_name + '_SW.npy'
    filename9 = args.expt_name + '_SB.npy'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.save(os.path.join(args.output_dir, filename4), nc1)
    np.save(os.path.join(args.output_dir, filename5), nc2)
    np.save(os.path.join(args.output_dir, filename6), nc3)
    np.save(os.path.join(args.output_dir, filename7), nc4)
    np.save(os.path.join(args.output_dir, filename8), sw)
    np.save(os.path.join(args.output_dir, filename9), sb)
    '''

    print("\nNC1 :", nc1)
    print("\nNC2 :", nc2)
    print("\nNC3 :", nc3)
    print("\nNC4 :", nc4)
    print("\nSW :", sw)
    print("\nSB :", sb)
    ## NC end //


    ## OOD detection eval (NINCO)
    print("\nOOD Dataset :", args.data_set)
    fpr, auroc = ood_evaluate(data_loader_val, data_loader_val2, model, linear_classifier, device, epoch, tf_writer=tf_writer, args=args)
    print("\nFPR:", 100*fpr)
    print("\nAUROC:", 100*auroc)

    ## save results
    filename2 = args.expt_name + '_' + args.data_set + '_fpr.npy'
    filename3 = args.expt_name + '_' + args.data_set + '_auroc.npy'
    np.save(os.path.join(args.output_dir, filename2), fpr)
    np.save(os.path.join(args.output_dir, filename3), auroc)
    print("\nFiles saved")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=100):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        trunc_normal_(self.linear.weight, std=0.01)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT linear probe training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.demo:
        args.batch_size = 50
        if 'visualize' in args.finetune:
            args.model, args.gate_scale, args.gate_center = 'avit_tiny_patch16_224', 5, 50
        elif 'tiny' in args.finetune:
            args.model = 'vit_tiny'
        elif 'small' in args.finetune:
            args.model = 'vit_small'
        else:
            print('Visualization of this model is not yet supported. Please modify the code and use only the function accordingly.')
            exit()

    main(args)
