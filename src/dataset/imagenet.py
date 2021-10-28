import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from RandAugment import RandAugment

import os
from torchvision.io import read_image
import json
from typing import Optional, Callable



def get_imagenet_transforms(args) :
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Lambda(lambda x: torch.cat([x, x, x], 0) if x.shape[0] == 1 else x),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Scale(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Lambda(lambda x: torch.cat([x, x, x], 0) if x.shape[0] == 1 else x),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    if args.randaugment :
        train_transform.transforms.insert(0, RandAugment(args.randaugment[0], args.randaugment[1]))
    return train_transform, test_transform


class Imagenet :
    def __init__(self, args, dataset='imagenet'):
        self.args = args

        if dataset == 'imagenet100' :
            Data = ImageNet100
            root = args.data_dir
        elif dataset == 'imagenet' :
            Data = ImageNet1K
            root = args.data_dir
        train_transform, test_transform = get_imagenet_transforms(args)
        
        train = Data(root=root, split='train', transform=train_transform)
        valid = Data(root=root, split='val', transform=test_transform)
        test = None
            
        if args.distributed:
            sampler_train = data.DistributedSampler(train)
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
        

class ImageNet1K(data.Dataset):
    def __init__(self, root: str, split: str = 'train', transform: Optional[Callable] = None):
        id_table_path = os.path.join(root+"labels.json")
        with open(id_table_path, "r") as id_table_json:
            id_table = json.load(id_table_json)
        self.id_list = os.listdir(root+split)
        self.id_str_list = list(id_table.values())
        self.sub_dir = os.path.join(root, f"{split}")
        self.img_dirs = sorted([img_dir for img_dir in os.listdir(self.sub_dir)])
        self.img_paths = sorted([os.path.join(self.sub_dir, img_dir, img_path) \
                                 for img_dir in os.listdir(self.sub_dir) \
                                 for img_path in os.listdir(os.path.join(self.sub_dir, img_dir))])
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_id = os.path.basename(os.path.dirname(img_path))
        label = self.id_list.index(img_id)
        
        image = read_image(img_path)
        image = transforms.ToPILImage()(image).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else :
            image = transforms.ToTensor()(image)

        return image, label


class ImageNet100(data.Dataset):
    def __init__(self, root: str, split: str = 'train', transform: Optional[Callable] = None):
        id_table_path = os.path.join(os.getcwd(), root+"labels.json")
        with open(id_table_path, "r") as id_table_json:
            id_table = json.load(id_table_json)
        self.id_list = list(id_table)
        self.id_str_list = list(id_table.values())
        self.sub_dir = os.path.join(os.getcwd(), root, f"{split}")
        self.img_dirs = sorted([img_dir for img_dir in os.listdir(self.sub_dir)])
        self.img_paths = sorted([os.path.join(self.sub_dir, img_dir, img_path) \
                                 for img_dir in os.listdir(self.sub_dir) \
                                 for img_path in os.listdir(os.path.join(self.sub_dir, img_dir))])
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_id = os.path.basename(os.path.dirname(img_path))
        label = self.id_list.index(img_id)
        
        image = read_image(img_path)
        image = transforms.ToPILImage()(image).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else :
            image = transforms.ToTensor()(image)

        return image, label
  
