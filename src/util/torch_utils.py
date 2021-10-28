

import torch, torchvision, torch_optimizer
import torch.optim as optim
from torch import Tensor
from typing import Optional, List
import numpy as np


########## NestedTensors

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)



# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


########## OPTIMIZATION

def get_optimizer(args, param_lr) :
    if args.optimizer == 'sgd' :
        optimizer = optim.SGD(
            param_lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw' :
        optimizer = optim.AdamW(
            param_lr,
            betas=(0.9,0.999),
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'lamb' :
        optimizer = torch_optimizer.Lamb(
            param_lr,
            betas=(0.9,0.999),
            weight_decay=args.weight_decay
        )
    else :
        raise ValueError(f"optimizer {args.optimizer} not supported")
    
    return optimizer


def get_scheduler(args, optimizer):

    if args.scheduler == 'steplr' :
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=list(args.lr_drop), 
            gamma=0.1
        )
    elif args.scheduler == 'cosine' : 
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=100,
            eta_min=0.0
        )
    else :
        raise ValueError(f"scheduler {args.scheduler} not supported")
    
    return scheduler


######## Define custom optimizers or schedulers here ###########




################################################################


# Utility Functions
@torch.no_grad()
def accuracy(input: Tensor,
             target: Tensor,
             top_k: int = 1
             ) -> Tensor:
    pred_idx = input.argmax(dim=-1, keepdim=True) if top_k == 1 else input.topk(k=top_k, dim=-1).indices
    target = target.view(-1, 1).expand_as(pred_idx)
    return (pred_idx == target).float().sum(dim=1).mean().item()

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(x, y, p=0.0):
    # code modified from https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
    if np.random.rand(1) < p :
        # generate mixed sample
        lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(x.size()[0]).to(x.device)
        y1 = y
        y2 = y[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        return x, y1, y2, lam
    return x, y, y, 1.0