from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import umap

from scipy.signal import savgol_filter
from six.moves import xrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from new_VQVAE_and_test.new_vqvae import Encoder, Decoder, Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------ Hyperparameters ------------------------------------ #
batch_size = 256
num_training_updates = 15000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512
learning_rate = 1e-3
commitment_cost = 0.25
decay = 0.99

# ------------------------------------- Data Settings ------------------------------------- #
training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

data_variance = np.var(training_data.data / 255.0)

training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)
validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)

model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# ----------------------------- Training & Information Output ----------------------------- #
model.train()
train_res_recon_error = []
train_res_perplexity = []
for i in xrange(num_training_updates):
    (data, _) = next(iter(training_loader))
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()
    
    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    if (i+1) % 100 == 0:
        print('%d iterations' % (i+1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()

# ---------------------------- Visualize Losses & Perplexities ---------------------------- #
train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)
f = plt.figure(figsize=(16,8))

ax = f.add_subplot(1,2,1)
ax.plot(train_res_recon_error_smooth)
ax.set_yscale('log')
ax.set_title('Smoothed NMSE.')
ax.set_xlabel('iteration')

ax = f.add_subplot(1,2,2)
ax.plot(train_res_perplexity_smooth)
ax.set_title('Smoothed Average codebook usage (perplexity).')
ax.set_xlabel('iteration')

ax.legend()
plt.show()

# ------------------------- Visualize Reconstructions & Originals ------------------------- #
def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()

model.eval()

(outcome_originals, _) = next(iter(validation_loader))
outcome_originals = outcome_originals.to(device)

vq_output_eval = model._pre_vq_conv(model._encoder(outcome_originals))
_, outcome_quantize, _, _ = model._vq_vae(vq_output_eval)
outcome_reconstructions = model._decoder(outcome_quantize)
show(make_grid(outcome_reconstructions.cpu().data)+0.5, )

(train_originals, _) = next(iter(training_loader))
train_originals = train_originals.to(device)
_, train_reconstructions, _, _ = model._vq_vae(train_originals)
show(make_grid(outcome_originals.cpu()+0.5))

# -------------------------------- Embedding Visualization -------------------------------- #
proj = umap.UMAP(n_neighbors=3,
                 min_dist=0.1,
                 metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())

plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
plt.legend()
plt.show()
