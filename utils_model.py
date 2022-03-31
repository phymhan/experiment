import torch
from torch import nn
from torch.nn import functional as F


def toggle_grad(model, on_or_off):
    if model is not None:
        for param in model.parameters():
            param.requires_grad = on_or_off

def requires_grad(model, flag=True):
    if model is not None:
        for p in model.parameters():
            p.requires_grad = flag

def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

""" e.g.
accum = 0.5 ** (32 / (10 * 1000))
ema_nimg = args.ema_kimg * 1000
if args.ema_rampup is not None:
    ema_nimg = min(ema_nimg, it * args.batch_size * args.ema_rampup)
accum = 0.5 ** (args.batch_size / max(ema_nimg, 1e-8))
accumulate(ema, module, 0 if args.no_ema else accum)
"""


# ========== Helper Module ==========
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class Reshape(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Reshape, self).__init__()

    def forward(self, input):
        return input.view(input.shape[0], -1)

class Squeeze(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Squeeze, self).__init__()

    def forward(self, input):
        return torch.squeeze(input)

class OneHot(nn.Module):
    def __init__(self, num_classes=10):
        super(OneHot, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, labels):
        device = labels.device
        labels = F.one_hot(labels, num_classes=self.num_classes).float().to(device)
        return labels

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Unflatten(nn.Module):
    def forward(self, x, size):
        return x.view(x.size(0), *size)
