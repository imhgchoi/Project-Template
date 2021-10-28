
from trainer.classifier_trainer import ClsTrainer

def build_trainer(args, dataset):
    if args.task_type == 'classification' :
        return ClsTrainer(args, dataset)
    else :
        raise ValueError(f'check task type : {args.task_type}')