import os
import sys
import shutil
from glob import glob
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import ast
import argparse
import torch

import pdb
st = pdb.set_trace


"""
helper classes
"""
class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __repr__(self):
        return str(self.__dict__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vec2sca_avg = 0
        self.vec2sca_val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if torch.is_tensor(self.val) and torch.numel(self.val) != 1:
            self.avg[self.count == 0] = 0
            self.vec2sca_avg = self.avg.sum() / len(self.avg)
            self.vec2sca_val = self.val.sum() / len(self.val)

""" e.g.
loss = AverageMeter()
loss.update(loss.mean().item(), n=batch_size)
print(loss.avg)
"""


# ========== random ==========
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========== args ==========
def str2bool(v):
    """
    borrowed from:
    https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
    :param v:
    :return: bool(v)
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(attr_bins):
    assert (isinstance(attr_bins, str))
    attr_bins = attr_bins.strip()
    if attr_bins.endswith(('.npy', '.npz')):
        attr_bins = np.load(attr_bins)
    else:
        assert (attr_bins.startswith('[') and attr_bins.endswith(']'))
        # attr_bins = np.array(ast.literal_eval(attr_bins))
        attr_bins = ast.literal_eval(attr_bins)
    return attr_bins


"""
logging helper functions assume args.log_dir is set
"""
# ========== logging ==========
def get_hostname():
    try:
        import socket
        return socket.gethostname()
    except:
        return 'unknown'

def print_args(parser, args, is_dict=False, flush=False):
    # args = deepcopy(args)  # NOTE
    if not is_dict and hasattr(args, 'parser'):
        delattr(args, 'parser')
    name = getattr(args, 'name', Path(args.log_dir).name if hasattr(args, 'log_dir') else 'unknown')
    datetime_now = datetime.now()
    message = f"Name: {name} Time: {datetime_now}\n"
    message += f"{os.getenv('USER')}@{get_hostname()}:\n"
    if os.getenv('CUDA_VISIBLE_DEVICES'):
        message += f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}\n"
    message += '--------------- Arguments ---------------\n'
    args_vars = args if is_dict else vars(args)
    for k, v in sorted(args_vars.items()):
        comment = ''
        default = None if parser is None else parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '------------------ End ------------------'
    if flush:
        print(message)

    # save to the disk
    log_dir = Path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir / 'src', exist_ok=True)
    file_name = log_dir / 'args.txt'
    with open(file_name, 'a+') as f:
        f.write(message)
        f.write('\n\n')

    # save command to disk
    file_name = log_dir / 'cmd.txt'
    with open(file_name, 'a+') as f:
        f.write(f'Time: {datetime_now}\n')
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            f.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
        f.write('deepspeed ' if getattr(args, 'deepspeed', False) else 'python3 ')
        f.write(' '.join(sys.argv))
        f.write('\n\n')

    # backup train code
    shutil.copyfile(sys.argv[0], log_dir / 'src' / f'{os.path.basename(sys.argv[0])}.txt')


# ========== wandb ==========
try:
    import wandb
except ImportError:
    wandb = None

import logging
def log(output, flush=True):
    logging.info(output)
    if flush:
        print(output)

def set_up_wandb_run_id(log_dir, resume=False):
    # NOTE: if resume, use the existing wandb run id, otherwise create a new one
    os.makedirs(log_dir, exist_ok=True)
    file_path = Path(log_dir) / 'wandb_run_id.txt'
    if resume:
        assert file_path.exists(), 'wandb_run_id.txt does not exist'
        with open(file_path, 'r') as f:
            run_id = f.readlines()[-1].strip()  # resume from the last run
    else:
        run_id = wandb.util.generate_id()
        with open(file_path, 'a+') as f:
            f.write(run_id + '\n')
    return run_id

def set_up_wandb(args):
    if wandb is not None:
        name = Path(args.log_dir).name
        resume = getattr(args, 'resume', False)
        run_id = set_up_wandb_run_id(args.log_dir, resume)
        args.wandb_run_id = run_id
        run = wandb.init(
            project=args.wandb_project,
            name=name,
            id=run_id,
            config=args,
            resume=True if resume else "allow",
        )
        return run
    else:
        log_str = "Failed to set up wandb - aborting"
        log(log_str, level="error")
        raise RuntimeError(log_str)


# ========== checkpointing ==========
def get_last_checkpoint(ckpt_dir, ckpt_ext='.pt', latest=None):
    assert ckpt_ext.startswith('.')
    if latest is None:
        ckpt_path = sorted(glob(os.path.join(ckpt_dir, '*'+ckpt_ext)), key=os.path.getmtime, reverse=True)[0]
    else:
        if not latest.endswith(ckpt_ext):
            latest += ckpt_ext
        ckpt_path = Path(ckpt_dir) / latest
    return ckpt_path


# ========== Network ==========
def toggle_grad(model, on_or_off):
    if model is not None:
        for param in model.parameters():
            param.requires_grad = on_or_off

def requires_grad(model, flag=True):
    if model is not None:
        for p in model.parameters():
            p.requires_grad = flag

# ========== Tensor and Numpy ==========
def randperm(n, ordered=False):
    if ordered:  # NOTE: whether to include ordered permutation
        return torch.randperm(n)
    else:
        perm_ord = torch.tensor(range(n))
        while True:
            perm = torch.randperm(n)
            if (perm != perm_ord).any():
                return perm        

def permute_dim(tensor, i=0, j=1, ordered=False):
    # Permute along dim i for each j.
    # e.g.: Factor-VAE, i = 0, j = 1; Jigsaw, i = 2, j = 0
    device = tensor.device
    n = tensor.shape[i]
    return torch.cat([torch.index_select(t, i, randperm(n, ordered).to(device)) for t in tensor.split(1, j)], j)
