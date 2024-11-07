from .ml_100k import ML100KDataset
from .beauty import BeautyDataset
from .games import GamesDataset
from .mooc_data import MoocDataset

DATASETS = {
    ML100KDataset.code(): ML100KDataset,
    BeautyDataset.code(): BeautyDataset,
    GamesDataset.code(): GamesDataset,
    MoocDataset.code(): MoocDataset
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)
