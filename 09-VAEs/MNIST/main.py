#%%
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from visdom import Visdom

from modules import VAE
from train_test import train
from train_test import test
import utils

#%%
log_interval = 100
seed = 1

torch.manual_seed(seed)

from chosen_gpu import get_freer_gpu
device = torch.device(get_freer_gpu())
print("Configured device: ", device)

#%%

out_dir = '../../../../data/torch_data/VGAN/MNIST/dataset' #you can use old downloaded dataset, I use from VGAN
batch_size=128

train_loader = torch.utils.data.DataLoader(datasets.MNIST(root=out_dir, download=True, train=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(root=out_dir, train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

#%%
# %%
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#%%

epochs = 50
viz = Visdom() 
global plotter, recon
plotter = utils.VisdomLinePlotter(env_name='main')
sample_image = utils.VisdomImage(env_name='main')
recon = utils.VisdomImage(env_name='main')

for epoch in range(1, epochs + 1):
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        print("save image: " + 'results/sample_' + str(epoch) + '.png')
        save_image(sample.view(64, 1, 28, 28), 'results/sample_' + str(epoch) + '.png')
        sample_image.display_image(sample.view(64, 1, 28, 28), 0, 'SAMPLE RECON')
    train(batch_size, epoch, model, train_loader, device, optimizer, plotter)
    test(batch_size, epoch, model, test_loader, device, optimizer, plotter, recon)