import time, os
import torch
import wandb

import util.general_utils as utils
from util.torch_utils import get_optimizer, get_scheduler
from models import build_model
from dataset import build_dataset
from trainer import build_trainer


def setup(args):
    utils.make_dirs()
    utils.init_distributed_mode(args)
    utils.set_seeds(args)


def main(args):
    # setup
    setup(args)
    device = torch.device(args.device)

    # model
    model = build_model(args).to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else :
        model_without_ddp = model
    
    # model params : customize if needed    
    params = set(model_without_ddp.parameters())
    param_n_lr = [{"params": list(params), "lr": args.lr}]

    # data : dataset contains train/val/test loaders 
    dataset = build_dataset(args)

    # optimization
    optimizer = get_optimizer(args, param_n_lr)
    scheduler = get_scheduler(args, optimizer)
    criterion = torch.nn.CrossEntropyLoss()   # TODO you should customize this based on your task

    # trainer
    trainer = build_trainer(args, dataset)

    # check for resume
    resume = False
    start_epoch = 0
    if args.resume :
        resume=True
        ckpt_path = f'out/{args.resume}/recent.pth'
        if not os.path.exists(ckpt_path) :
            raise FileNotFoundError
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_checkpoint['scheduler']
        trainer.best_key_metric = checkpoint['best_key_metric']
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
         
    # set wandb 
    if args.use_wandb and utils.get_rank() == 0 :
        wandb.init(entity=args.wandb_entity, 
                   project=args.wandb_pname,
                   name=args.run_name,
                   id=args.run_id,
                   resume=resume)
        wandb.config.update(args)
        wandb.watch(model)
    
    # train
    trainer.train(model=model,
                  model_without_ddp=model_without_ddp,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  criterion=criterion,
                  start_epoch=start_epoch) 

    wandb.finish()


if __name__ == "__main__":
    from config import get_args
    args = get_args()
    if args.use_wandb :
        args.run_id = args.run_name + str(time.time()).split('.')[-1]

    main(args)