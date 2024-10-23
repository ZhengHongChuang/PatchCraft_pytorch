from modules.net import get_model, get_loss_fn, get_optimizer
from data.datasets import TrainDataset
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import torch
import os
import torch.distributed as dist
from utils.utils import init_distributed_mode, get_world_size, get_rank
import torch.backends.cudnn as cudnn
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

def args_parser():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    parser.add_argument("--data_path",type=str,default="/home/ubuntu/datasets/GenImages/sd5",help="path to the data")
    parser.add_argument("--log_dir", type=str, default="logs_ddp", help="path to the logs")
    parser.add_argument("--outputs", type=str, default="outputs_ddp", help="path to the outputs")
    parser.add_argument("--use_ddp", type=bool, default=True, help="Use DataParallel for multi-GPU training")
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument("--num_works", type=int, default=8, help="total number of processes")
    return parser.parse_args()

def evaluate(net, dataloader, criterion, device, writer=None, epoch=None):
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Evaluating')
        for batch_idx, (rich_texture, poor_texture, label) in progress_bar:
            rich_texture, poor_texture, label = (
                rich_texture.to(device).float(),
                poor_texture.to(device).float(),
                label.to(device).unsqueeze(1).float())
            output = net(rich_texture, poor_texture)
            loss = criterion(output, label)
            total_loss += loss.item()

            predicted = (output >= 0.5).float()
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

    train_datasets = TrainDataset(args.data_path, is_train=True)
    test_datasets = TrainDataset(args.data_path, is_train=False)
    
    train_sampler = DistributedSampler(train_datasets, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
    test_sampler = DistributedSampler(test_datasets, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)

    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, sampler=train_sampler,num_workers=args.num_works)
    test_dataloader = DataLoader(test_datasets, batch_size=args.batch_size, sampler=test_sampler,num_workers=args.num_works)

    writer = SummaryWriter(args.log_dir) if args.rank == 0 else None
    os.makedirs(args.outputs, exist_ok=True)

    net = get_model().to(args.device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=True)
    criterion = get_loss_fn().to(args.device)
    optimizer = get_optimizer(net, args.lr)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch [{epoch + 1}/{args.epochs}]')
        net.train()
        total_loss = 0.0
        for batch_idx, (rich_texture, poor_texture, label) in progress_bar:
            rich_texture, poor_texture, label = (
                rich_texture.to(args.device).float(),
                poor_texture.to(args.device).float(),
                label.to(args.device).unsqueeze(1).float())
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
                    writer.add_scalar("avg_loss", avg_loss, epoch * len(train_dataloader) + batch_idx)

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

        test_loss, test_accuracy = evaluate(net, test_dataloader, criterion, args.device, writer, epoch)
        if writer:
            writer.add_scalar("test_loss", test_loss, epoch)
            writer.add_scalar("test_accuracy", test_accuracy, epoch)

    if writer:
        writer.close()
    dist.destroy_process_group()



if __name__ == "__main__":
    args = args_parser()
    train(args)
