import torch
import os
from util.general_utils import save_on_master

class BaseTrainer :
    def __init__(self, args):
        self.args = args

    def save(self, model_without_ddp, optimizer, scheduler, best, epoch, filename):
        if self.args.save :
            out_folder = f'out/{self.args.run_name}/'
            if not os.path.exists(out_folder) :
                os.mkdir(out_folder)
            checkpoint_path = out_folder + f'{filename}.pth'
            save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_key_metric' : best,
                'epoch': epoch,
                'args': self.args,
            }, checkpoint_path)

    def train(self):
        raise NotImplementedError('train function not implemented')

    @torch.no_grad()
    def evaluate(self, model, dataloader):
        raise NotImplementedError('evaluate function not implemented')