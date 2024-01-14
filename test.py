#from torchsummary import summary
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Normalize, Compose, ToTensor
from tqdm import tqdm
from torch.cuda.amp import autocast
from torchvision.utils import make_grid
from torchvision import transforms
import matplotlib.pyplot as plt
from models import vqvae
# from dataset import TinyImageNet
# from dataset import TrainTinyImageNet, ValTinyImageNet,load_tinyimagenet
from dataset import TinyImageNet
import torch
import numpy as np
import os
import cv2
import argparse

vae = vqvae.VQVAE(1, 28, 100, [16, 64, 128]).to('cuda')
transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

# trainset = MNIST(root='./', download=True, transform=transform, train=True)
data_dir = './tiny-imagenet-200/'
trainset = TinyImageNet(data_dir, train=True)
dataloader=DataLoader(trainset, batch_size=512)
optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3)

tqdm_bar=tqdm(range(5))
for ep in tqdm_bar:
    for i, (x, _) in enumerate(dataloader):
        x = x.to('cuda').float()
        with autocast():
            recon, input, vq_loss=vae(x)
            loss=vae.loss_function(recon, input, vq_loss)
        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        if i%10==0:
            tqdm_bar.set_description('loss: {}'.format(loss['loss']))

# dataloader_test = DataLoader(MNIST(root='./', download=True, transform=transform, train=False), batch_size=16, shuffle=True)
testset = TinyImageNet(data_dir, train=False)
dataloader_test=DataLoader(testset, batch_size=16, shuffle=True)
# vae.load_state_dict(torch.load('./vae.pt'))
vae.eval()
for x,_ in dataloader_test:  
    x=x.to('cuda').float()
    reconstruct_x=vae.generate(x)
    # 前两行为输入数字
    # 后两行为重建数字
    new_x=torch.cat([x, reconstruct_x.detach()], dim=0)
    grid_pics=make_grid(new_x.to('cpu'), 8)
    grid_pics = grid_pics.clamp(0,255)
    plt.imshow(grid_pics.permute(1,2,0))
    plt.show()
    break