import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from RandAugment import RandAugment



def get_cifar_transforms(args):
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]
    normalize = transforms.Normalize(mean, std)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    if args.randaugment :
        train_transform.transforms.insert(0, RandAugment(args.randaugment[0], args.randaugment[1]))
        
    return train_transform, valid_transform


class CIFAR :
    def __init__(self, args, dataset='cifar10'):
        self.args = args
        
        Data = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
        root = f'data/{dataset}/' if args.data_dir is None else args.data_dir
        train_transform, valid_transform = get_cifar_transforms(args)
        train = Data(root=root, 
                    train=True, 
                    transform=train_transform, 
                    download=True)
        valid = Data(root=root, 
                    train=False, 
                    transform=valid_transform, 
                    download=True)
        test = None
            
        if args.distributed:
            sampler_train = data.DistributedSampler(train, shuffle=True)
            sampler_valid = data.DistributedSampler(valid, shuffle=False)
        else:
            sampler_train = data.RandomSampler(train)
            sampler_valid = data.SequentialSampler(valid)
        self.sampler_train = sampler_train
        self.sampler_valid = sampler_valid
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        self.train_loader = data.DataLoader(train, 
                                            batch_sampler=batch_sampler_train,
                                            pin_memory=False,
                                            num_workers=args.num_workers)
        self.valid_loader = data.DataLoader(valid,
                                            batch_size=args.batch_size,
                                            sampler=sampler_valid,
                                            drop_last=False,
                                            pin_memory=False,
                                            num_workers=args.num_workers)
        self.test_loader = None
        
    