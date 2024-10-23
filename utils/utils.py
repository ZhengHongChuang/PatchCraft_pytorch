import os
import torch
import torch.distributed as dist


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
