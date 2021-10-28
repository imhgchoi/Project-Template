import argparse


def get_args():

    parser = argparse.ArgumentParser('Project Template', add_help=False)

    # PROJECT
    parser.add_argument('--run_name', default='test', type=str)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_pname", type=str, default='your_project_name')
    parser.add_argument("--wandb_entity", type=str, default='your_wandb_id')
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=('classification','regression','detection','rl'))
    parser.add_argument("--save", action="store_true")
    parser.add_argument('--resume', type=str)

    # DIRECTORIES
    parser.add_argument('--data_dir', type=str, required=True)

    # DATA CONFIG
    parser.add_argument('--dataset', type=str, 
                        choices=('cifar10','cifar100','imagenet100','imagenet1k','coco2017'))
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--randaugment", type=int, nargs='+',
                        help="--randaugment (layers) (intensity)")

    # MODEL CONFIG
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dropout", type=float, default=0.0)

    # OPTIMIZATION CONFIG
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=('sgd','adamw'))
    parser.add_argument("--scheduler", type=str, default='steplr',
                        choices=('steplr','cosine'))
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=[200], type=int, nargs='+')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_grad', default=0.1, type=float)

    
    # RESOURCE
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, 
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')

    # MISCELLANEOUS
    parser.add_argument('--debug', action='store_true')
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--cutmix_prob", type=float, default=0.0)

    return parser.parse_args()
    
    