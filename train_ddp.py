from data.datasets import TrainDataset, TestDataset
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import torch
import os
import torch.distributed as dist
from utils.utils import init_distributed_mode, get_world_size, adjust_learning_rate, get_rank, load_model, load_loss_fn, load_optimizer, use_data_parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
import warnings

warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module="torch.functional")


def args_parser():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="batch size")
    # parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument('--T_max',
                        type=int,
                        default=10,
                        help='Maximum number of epochs for cosine annealing')
    parser.add_argument('--min_lr',
                        type=float,
                        default=0.001,
                        help='Minimum learning rate')
    parser.add_argument('--max_lr',
                        type=float,
                        default=0.1,
                        help='Maximum learning rate')
    parser.add_argument('--warmup_epochs',
                        type=int,
                        default=5,
                        help='Number of warmup epochs')
    parser.add_argument("--epochs",
                        type=int,
                        default=100,
                        help="number of epochs")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="device to use")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/ubuntu/datasets/facedatasets",
        # default="/home/ubuntu/train/PatchCraft_pytorch/datasets",
        help="path to the data")
    parser.add_argument(
        "--test_path",
        type=str,
        # default="/home/ubuntu/datasets/facedatasets",
        default="/home/ubuntu/train/PatchCraft_pytorch/datasets",
        help="path to the data")
    # parser.add_argument("--data_path",type=str,default="/home/ubuntu/datasets/GenImages/sd5",help="path to the data")
    # parser.add_argument("--data_path", type=str, default="datasets", help="path to the data")
    parser.add_argument("--log_dir",
                        type=str,
                        default="logs_ddp4",
                        help="path to the logs")
    parser.add_argument("--outputs",
                        type=str,
                        default="outputs_ddp4",
                        help="path to the outputs")
    parser.add_argument(
        "--use_ddp",
        type=bool,
        default=True,
        help="Use DistributedDataParallel for multi-GPU training")
    parser.add_argument("--use_dp",
                        type=bool,
                        default=False,
                        help="Use DataParallel for multi-GPU training")
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument("--num_works",
                        type=int,
                        default=8,
                        help="total number of processes")
    parser.add_argument("--resume",
                        type=str,
                        default=None,
                        help="path to the checkpoint")
    return parser.parse_args()


def evaluate(net, dataloader, criterion, device, writer=None, epoch=None):
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader),
                            total=len(dataloader),
                            desc='Evaluating')
        for batch_idx, (rich_texture, poor_texture, label) in progress_bar:
            rich_texture, poor_texture, label = (
                rich_texture.to(device).float(),
                poor_texture.to(device).float(), label.to(device).long()
            )

            output = net(rich_texture, poor_texture)

            loss = criterion(output, label)
            total_loss += loss.item()

            # 使用 torch.argmax 来获得预测类别
            predicted = torch.argmax(output, dim=1)  # output 为 logits
            correct += (predicted == label).sum().item()
            total += label.size(0)

            progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    if writer is not None and epoch is not None:
        writer.add_scalar("test_loss", avg_loss, epoch)
        writer.add_scalar("test_accuracy", accuracy, epoch)

    return avg_loss, accuracy



def train(args):
    init_distributed_mode(args)
    cudnn.benchmark = True

    datasets = TrainDataset(args.data_path)
    train_datasets , test_datasets = random_split(datasets, [int(0.8*len(datasets)), len(datasets)-int(0.8*len(datasets))])
    # test_datasets = TestDataset(args.test_path)

    train_sampler = DistributedSampler(train_datasets,
                                       num_replicas=get_world_size(),
                                       rank=get_rank(),
                                       shuffle=True)
    test_sampler = DistributedSampler(test_datasets,
                                      num_replicas=get_world_size(),
                                      rank=get_rank(),
                                      shuffle=False)

    train_dataloader = DataLoader(train_datasets,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=args.num_works)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=args.batch_size,
                                 sampler=test_sampler,
                                 num_workers=args.num_works)

    writer = SummaryWriter(args.log_dir) if args.rank == 0 else None
    os.makedirs(args.outputs, exist_ok=True)

    net = load_model(args)
    net = use_data_parallel(net, args)
    criterion = load_loss_fn(args)
    optimizer = load_optimizer(net, args.min_lr)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader),
                            desc=f'Epoch [{epoch + 1}/{args.epochs}]')
        net.train()
        total_loss = 0.0
        for batch_idx, (rich_texture, poor_texture, label) in progress_bar:
            rich_texture, poor_texture, label = (rich_texture.to(
                args.device).float(), poor_texture.to(args.device).float(),
                                                 label.to(args.device).long())
            optimizer.zero_grad()
            output = net(rich_texture, poor_texture)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_loss = total_loss / (batch_idx + 1)
                if writer:
                    writer.add_scalar(
                        "avg_loss", avg_loss,
                        epoch * len(train_dataloader) + batch_idx)
        adjust_learning_rate(optimizer,epoch,args)
        avg_loss = total_loss / len(train_dataloader)

        if writer:
            writer.add_scalar("avg_loss_epoch", avg_loss, epoch)

        if args.rank == 0:
            checkpoint = {
                'model': net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, f"{args.outputs}/checkpoint_{epoch}.pth")

        test_loss, test_accuracy = evaluate(net, test_dataloader, criterion,
                                            args.device, writer, epoch)
        if writer:
            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("test_accuracy", test_accuracy, epoch)

    if writer:
        writer.close()
    dist.destroy_process_group()


def print_os():
    print("|| MASTER_ADDR:", os.environ["MASTER_ADDR"], "|| MASTER_PORT:",
          os.environ["MASTER_PORT"], "|| LOCAL_RANK:",
          os.environ["LOCAL_RANK"], "|| RANK:", os.environ["RANK"],
          "|| WORLD_SIZE:", os.environ["WORLD_SIZE"])


if __name__ == "__main__":
    print_os()
    args = args_parser()
    train(args)
