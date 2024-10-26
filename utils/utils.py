import os
import math
import torch
import torch.distributed as dist
import torch.nn as nn
from modules.net import RichPoorTextureContrastModel
from torch.utils.data.distributed import DistributedSampler
from data.datasets import TrainDataset
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if args.use_ddp:
        args.rank = int(os.environ.get('RANK', 0))
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        args.gpu = int(os.environ.get('LOCAL_RANK', 0))
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'],
                                         os.environ['MASTER_PORT'])

        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'

    print('| distributed init (local_rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu),
          flush=True)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        rank=args.rank,
        world_size=args.world_size
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def load_model(args):
    return RichPoorTextureContrastModel().to(args.device)


def load_loss_fn(args):
    return nn.CrossEntropyLoss().to(args.device)
    # return nn.BCELoss().to(args.device)


def load_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


def load_checkpoint(args, net):
    checkpoint = torch.load(args.net_pth)
    net.load_state_dict(checkpoint['model'])
    return checkpoint['epoch']

def set_DistributedSampler(datasets,shuffle=True):
    datasets_sampler = DistributedSampler(datasets,
                                          num_replicas=get_world_size(),
                                          rank=get_rank(),
                                          shuffle=shuffle)
    return datasets_sampler

def get_datasets(args,is_train=True):
    datasets = TrainDataset(args.data_path, is_train=is_train)
    return datasets

def get_dataloader(datasets, sampler, args):
    dataloader = DataLoader(datasets, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_works)
    return dataloader


def set_logs(args):
    if args.rank == 0:
        writer = SummaryWriter(args.log_dir)
        os.makedirs(args.outputs, exist_ok=True)
    else:
        writer = None
    return writer

def set_dataParallel(args,net):
    if args.use_dp:
        net = torch.nn.DataParallel(net)
    elif args.use_ddp:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)
    return net


def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.max_lr * (epoch + 1) / args.warmup_epochs
    else:
        lr = args.min_lr + (args.max_lr - args.min_lr) * (
            1 + math.cos(math.pi * (epoch - args.warmup_epochs) /
                         (args.T_max - args.warmup_epochs))) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
