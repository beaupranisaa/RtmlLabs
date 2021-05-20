#%%
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torchvision
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

compose = transforms.Compose(
    [
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        #transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

ds = torchvision.datasets.ImageFolder(root='dataset/', transform=compose)

ratio = [int(len(ds)*0.98), len(ds) - int(len(ds)*0.98)]

train_dataset, test_dataset = torch.utils.data.random_split(ds, ratio)

batch_size=4

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False,num_workers=1, pin_memory=True)
print('train_loader', len(train_loader))
print('test_loader', len(test_loader))

#%%
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas = (0.5,0.999), weight_decay = 0.0005)

#%%

epochs = 100
viz = Visdom() 
global plotter, recon
plotter = utils.VisdomLinePlotter(env_name='main')
sample_image = utils.VisdomImage(env_name='main')
recon = utils.VisdomImage(env_name='main')

for epoch in range(1, epochs + 1):
    with torch.no_grad():
        sample = torch.randn(32,32).to(device)
        sample = model.decode(sample).cpu()
        print("save image: " + 'results/sample_' + str(epoch) + '.png')
        save_image(sample, 'results/sample_' + str(epoch) + '.png')
        sample_image.display_image(sample, 0, 'SAMPLE RECON')
    train(batch_size, epoch, model, train_loader, device, optimizer, plotter)
    test(batch_size, epoch, model, test_loader, device, optimizer, plotter, recon)