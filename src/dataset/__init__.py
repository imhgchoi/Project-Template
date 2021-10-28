


from dataset.cifar import CIFAR
from dataset.imagenet import Imagenet
from dataset.coco import COCO


def build_dataset(args):
    dataset = args.dataset
    if dataset in ('cifar10', 'cifar100') :
        data = CIFAR(args, dataset)
    elif dataset in ('imagenet100', 'imagenet'):
        data = Imagenet(args, dataset)
    elif dataset in ('coco2017') :
        data = COCO(args, dataset)    
    else :
        raise ValueError(f"dataset {dataset} not supported")
    
    if args.debug :
        data.train_loader = data.valid_loader if data.test_loader is None else data.test_loader
        
    return data

