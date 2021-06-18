from os import name
import torch
import torch.nn as nn
import random
import numpy as np
import optuna
from torch.utils.data.dataloader import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .data_loader.data_loaders import loader
from .data_loader.EventDataset import EventDataset


def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def collate_fn(batch):
    return tuple(zip(*batch))
    
def objective(trial: optuna.Trial):
    params = {}
    batch_size = 16

    torch.manual_seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

    train_set = []
    validate_dataloaders = {}
    test_dataloaders = {}
    for dataset in datasets:
        train, test, validate = loader(dataset)
        train_set.extend(train)
        validate_dataloader = DataLoader(EventDataset(validate), batch_size=batch_size, shuffle=True,collate_fn=collate_fn, worker_init_fn=seed_worker)
        test_dataloader = DataLoader(EventDataset(test), batch_size=batch_size, shuffle=True,collate_fn=collate_fn, worker_init_fn=seed_worker)
        validate_dataloaders[dataset] = validate_dataloader
        test_dataloaders[dataset] = test_dataloader
    train_dataloader = DataLoader(EventDataset(train_set), batch_size=batch_size, shuffle=True,collate_fn=collate_fn, worker_init_fn=seed_worker)
    

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='SEED', default=0, type=int)
    parser.add_argument('--dataset', help="Name of dataset", action='append', required=True)

    args = parser.parse_args()
    seed = args.seed
    datasets = args.dataset


