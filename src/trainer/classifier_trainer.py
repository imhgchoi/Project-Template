import torch
import wandb

from trainer.base_trainer import BaseTrainer
import util.general_utils as utils
from util.torch_utils import accuracy, cutmix

class ClsTrainer(BaseTrainer):
    def __init__(self, args, dataset):
        super().__init__(args)
        # arguments
        self.epochs = args.epochs
        self.device = args.device

        self.clip_grad = args.clip_grad
        self.cutmix_prob = 0.0

        # data
        self.tr_loader = dataset.train_loader
        self.vl_loader = dataset.valid_loader
        self.te_loader = dataset.test_loader

        self.base_ds = dataset.sampler_train

        # report
        self.print_freq = args.print_every
        self.save_freq = args.save_every
        self.best_key_metric = 0.0


    def train(self, model, model_without_ddp, optimizer, scheduler, criterion, start_epoch):
        
        for epoch in range(start_epoch, self.epochs):
            if self.args.distributed :
                self.base_ds.set_epoch(epoch)
                
            # train phase
            model, optimizer, scheduler, train_stats = self.train_one_epoch(model, 
                                                                            optimizer, 
                                                                            scheduler, 
                                                                            criterion,
                                                                            epoch)
            
            # test phase
            valid_stats = self.evaluate(model, self.vl_loader, criterion, epoch)
            
            # resolve
            if utils.get_rank() == 0 :
                if valid_stats['accuracy'] > self.best_key_metric  :
                    self.best_key_metric = valid_stats['accuracy']
                    self.save(model_without_ddp, 
                              optimizer, 
                              scheduler, 
                              self.best_key_metric,
                              epoch, 
                              filename='best')
                if epoch % self.save_freq == self.save_freq-1 :
                    self.save(model_without_ddp, 
                              optimizer, 
                              scheduler, 
                              self.best_key_metric,
                              epoch, 
                              filename='recent')  
                
                if self.args.use_wandb : 
                    wandb_log = {
                        "Train Accuracy": train_stats['accuracy'],
                        "Train Top5 Accuracy" : train_stats['accuracy_top5'],
                        "Train Loss": train_stats['loss'],
                        "Valid Accuracy": valid_stats['accuracy'],
                        "Valid Top5 Accuracy" : valid_stats['accuracy_top5'],
                        "Valid Loss": valid_stats['loss']
                    }
                    wandb.log(wandb_log)
            print('-'*100)
        
        if self.te_loader is not None :
            self.evaluate(model, self.te_loader, criterion)


    def train_one_epoch(self, model, optimizer, scheduler, criterion, epoch):
        model.train()
        criterion.train()
        
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('accuracy_top5', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.3f}'))
        header = f'\nEpoch {epoch} Train'
        
        for X, y in metric_logger.log_every(self.tr_loader, self.print_freq, header):
            X = X.to(self.device)
            y = y.to(self.device)
            X, y1, y2, lam = cutmix(X, y, p=self.cutmix_prob)
            
            logits = model(X)
            loss = criterion(logits, y1) * lam + criterion(logits, y2) * (1. - lam)
            batch_acc = accuracy(logits, y)
            batch_acc_top5 = accuracy(logits, y, top_k=5)
            
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm = 0
            for name, p in model.named_parameters():
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += param_norm ** 2
            grad_norm = grad_norm ** 0.5
            
            stat_dict = {'loss': loss, 
                         'grad_norm': grad_norm,
                         'accuracy': torch.Tensor([batch_acc])[0].to(loss.device),
                         'accuracy_top5': torch.Tensor([batch_acc_top5])[0].to(loss.device)}
            stat_dict_reduced = utils.reduce_dict(stat_dict)
            loss_value = stat_dict_reduced['loss'].item()
            
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad)
            optimizer.step()
            
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[-1]["lr"])
            metric_logger.update(accuracy=stat_dict_reduced['accuracy'] * 100)
            metric_logger.update(accuracy_top5=stat_dict_reduced['accuracy_top5'] * 100)
            metric_logger.update(grad_norm=stat_dict_reduced['grad_norm'])

        metric_logger.synchronize_between_processes()
        scheduler.step()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        print(f"\n\tAveraged stats - accuracy (acc top5): {stats['accuracy']:.2f} ({stats['accuracy_top5']:.2f})  "
              + f"loss: {stats['loss']:.4f} ")

        return model, optimizer, scheduler, stats

    @torch.no_grad()
    def evaluate(self, model, dataloader, criterion, epoch='') :
        model.eval()
        criterion.eval()

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('accuracy', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('accuracy_top5', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = f'\nEPOCH {epoch} Test'

        with torch.no_grad() :
            for X, y in metric_logger.log_every(dataloader, self.print_freq, header):
                X = X.permute(0,2,3,1).to(self.device)
                y = y.to(self.device)
                
                logits = model(X)
                loss = criterion(logits, y)
                batch_acc = accuracy(logits, y)
                batch_acc_top5 = accuracy(logits, y, top_k=5)
                
                stat_dict = {'loss': loss, 
                             'accuracy': torch.Tensor([batch_acc])[0].to(loss.device),
                             'accuracy_top5': torch.Tensor([batch_acc_top5])[0].to(loss.device)}
                stat_dict_reduced = utils.reduce_dict(stat_dict)
                loss_value = stat_dict_reduced['loss'].item()
                
                metric_logger.update(loss=loss_value)
                metric_logger.update(accuracy=stat_dict_reduced['accuracy']*100)
                metric_logger.update(accuracy_top5=stat_dict_reduced['accuracy_top5']*100)
                     
            metric_logger.synchronize_between_processes()
            stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            print(f"\n\tAveraged stats - accuracy (acc top5): {stats['accuracy']:.2f} ({stats['accuracy_top5']:.2f})  "
                  + f"loss: {stats['loss']:.4f}")

        return stats
        