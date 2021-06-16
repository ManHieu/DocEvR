from os import name
import torch
import torch.nn as nn
import random
import numpy as np
import optuna
from torch.utils.data.dataloader import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .data_loader.data_loaders import loader


def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
def objective(trial: optuna.Trial):
    torch.manual_seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

    train_set = []
    for dataset in datasets:
        train, test, valid = loader(dataset)

    

    params = {}


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='SEED', default=0, type=int)
    parser.add_argument('--dataset', help="Name of dataset", action='append', required=True)

    args = parser.parse_args()
    seed = args.seed
    datasets = args.dataset


