
from models.sample import SampleModel


def build_model(args):
    if args.model == 'sample':
        return SampleModel(args)
    else :
        raise NotImplementedError(f"check model name : {args.model}")