import torch
import torchvision.transforms as T


normalize = T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
single_transform = T.Compose([T.ToTensor(), normalize])


# ========== transforms ==========
class BatchwiseTransform:
    def __init__(self, transform):
        # perform random transform along batch dimension
        self.transform = transform

    def __call__(self, x):
        # x: [B, C, H, W]
        y = [self.transform(i) for i in x]
        return torch.stack(y, dim=0)


# ========== data sampler ==========
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)
""" e.g.
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
    drop_last=True,
    num_workers=args.num_workers,
)
"""

""" limit train batches
indices = torch.randperm(len(dataset))[:int(args.limit_train_batches * len(dataset))]
dataset = torch.utils.data.Subset(dataset, indices)
"""
