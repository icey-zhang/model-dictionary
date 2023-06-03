import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from nets_H import FusionNet, UncertaintyNet
import scipy.io as sio
from sklearn.preprocessing import scale
from datasets_H import matDataset
from metrics import cluster, classification, tsne, UMAP

# os.environ['CUDA_VISIBLE_DEVICES'] = '0，1'

def train():

    data = sio.loadmat('cub_googlenet_doc2vec_c10.mat')
    Views = len(data['X'][0])
    N = len(data['X'][0][0])#len(data['X'][0][0].T)
    Y = data['gt']-1
    dataset = data['X'][0]
    X = []
    fu_nets = []
    sigmas = []
    for v in range(Views):
        data_tmp = data['X'][0][v]#.T
        data_tmp = scale(data_tmp)
        data_tmp = torch.from_numpy(data_tmp).float().to(device)
        X.append(data_tmp)
        net_tmp = FusionNet(args.dim, data_tmp.shape[1]).to(device)
        net_tmp = torch.nn.DataParallel(net_tmp)
        fu_nets.append(net_tmp)
        sigma_tmp = UncertaintyNet(args.dim, data_tmp.shape[1]).to(device)
        sigmas.append(sigma_tmp)
    dataset = matDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    H = torch.normal(mean=torch.zeros([N, args.dim]), std=0.01).to(device).detach()
    H.requires_grad_(True)
    op = optim.Adam(chain(nn.ModuleList(fu_nets).parameters(),nn.ModuleList(sigmas).parameters(), [H]), lr=args.lr[1])
    lr_s = torch.optim.lr_scheduler.StepLR(op, step_size=20, gamma=0.9)

    op_pre = optim.Adam(chain(nn.ModuleList(fu_nets).parameters(), [H]), lr=args.lr[0])

    for epoch_pre in range(args.epochs[0]):
        for batch_idx, (batch, y, idx) in enumerate(train_loader):
            pre_loss = 0
            for v in range(Views):
                H_batch = H[idx]
                x_re = fu_nets[v](H_batch)
                re_loss = (x_re - batch[v])**2
                pre_loss = pre_loss + re_loss.mean()
            op_pre.zero_grad()
            pre_loss.backward()
            op_pre.step()

            if batch_idx % args.log_interval == 0:
                print('Pretrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_pre, batch_idx * len(y), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), pre_loss))

    for epoch in range(args.epochs[1]):
        for batch_idx, (batch, y, idx) in enumerate(train_loader):
            loss = 0
            for v in range(Views):
                H_batch = H[idx]
                x_re = fu_nets[v](H_batch)
                sigma = sigmas[v](H_batch)
                re_loss = (x_re - batch[v])**2
                re_loss = re_loss.mul(0.5 * torch.exp(-sigma)) + 0.5 * sigma
                re_loss = re_loss.mean(1, keepdim=True)
                loss = loss + re_loss.mean()
            op.zero_grad()
            loss.backward()
            lr_s.step()
            op.step()
            if batch_idx % args.log_interval == 0:
                print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(y), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss))
    return H.detach().cpu().numpy(), Y


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n-latents', type=int, default=100,
                        help='size of the latent embedding [default: 64]')
    parser.add_argument('--batch-size', type=int, default=2000, metavar='N',
                        help='input batch size for training [default: 2000]')
    parser.add_argument('--epochs', type=int, default=[200, 100], metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--lr', type=float, default=[5e-3, 1e-3], metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--dim', type=float, default=20,
                        help='hidden dimension for fusion net [default: 20]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: False]')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='choose CUDA device [default: cuda:1]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    device = torch.device(args.device if args.cuda else 'cpu')
    times = 30
    acc = np.zeros([times])
    p = np.zeros([times])
    r = np.zeros([times])
    f1 = np.zeros([times])
    for t in range(times):
        H, Y = train()
        acc[t], _, p[t], _, r[t], _, f1[t], _ = classification(H, Y.squeeze(), count=1, test_size=0.2, rt=True)
        # acc[t], _, nmi[t], _, RI[t], _, f1[t], _ = cluster(len(np.unique(Y)), H, Y.squeeze(), count=10, test_size=0.2, rt=True)
    print(f'{times} times ACC = {acc.mean():.2f}±{acc.std():.2f}\n'
          f'p = {p.mean():.2f}±{p.std():.2f}\n'
          f'r = {r.mean():.2f}±{r.std():.2f}\n'
          f'f1 = {f1.mean():.2f}±{f1.std():.2f}')