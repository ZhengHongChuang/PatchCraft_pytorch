from modules.net import get_model, get_loss_fn, get_optimizer
from data.datasets import TrainDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import torch
import os

def args_parser():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    parser.add_argument("--data_path", type=str, default="imgs", help="path to the data")
    parser.add_argument("--log_dir", type=str, default="logs", help="path to the logs")
    parser.add_argument("--outputs", type=str, default="outputs", help="path to the outputs")
    return parser.parse_args()
def train(args):
    writer = SummaryWriter(args.log_dir)
    if args.outputs:
        os.makedirs(args.outputs, exist_ok=True)
    train_datasets = TrainDataset(args.data_path)
    dataloader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)  
    net = get_model().to(args.device)
    criterion = get_loss_fn().to(args.device)
    optimizer = get_optimizer(net, args.lr)
    for epoch in range(args.epochs):
        progress_bar = tqdm(enumerate(dataloader),
                                total=len(dataloader),
                                desc=f'Epoch [{epoch+1}/{args.epochs}]',)
        for batch_idx , (rich_texture, poor_texture, label) in progress_bar:
            rich_texture, poor_texture, label = rich_texture.to(args.device).float(), poor_texture.to(args.device).float(), label.to(args.device).unsqueeze(1).float()
            optimizer.zero_grad()
            output = net(rich_texture, poor_texture)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss':loss.item()})
            if batch_idx % 10 == 0:
                writer.add_scalar("loss", loss.item(), epoch*len(dataloader)+batch_idx)
        checkpoint = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, f"{args.outputs}/checkpoint_{epoch}.pth")
    writer.close()
        

if __name__=="__main__":
    args = args_parser()
    train(args)