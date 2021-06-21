import datetime
from exp import EXP
from utils.constant import CUDA
from models.predictor_model import ECIRobertaJointTask
from models.selector_model import LSTMSelector, SelectorModel
import torch
import torch.nn as nn
import random
import numpy as np
import optuna
from torch.utils.data.dataloader import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from data_loader.loader import loader
from data_loader.EventDataset import EventDataset


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def collate_fn(batch):
    return tuple(zip(*batch))
    
def objective(trial: optuna.Trial):
    params = {
        's_hidden_dim': 512,
        's_mlp_dim': 512,
        'p_mlp_dim': 512, 
        'n_head': 12,
        "epoches": 5,
        "task_weights": {
            '1': 1, # 1 is HiEve
            '2': 1, # 2 is MATRES.
            # '3': trial.suggest_float('I2B2_weight', 0.4, 1, step=0.2),
        },
        'num_ctx_select': 3,
        's_lr': 1e-5,
        'p_lr': 3e-7,

    }
    batch_size = 2
    drop_rate = 0.5
    fn_activative = 'relu6'
    is_mul = True
    # trial.suggest_categorical('is_mul', [True, False])
    is_sub = True
    # trial.suggest_categorical('is_sub', [True, False])

    torch.manual_seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

    train_set = []
    train_short_set = []
    validate_dataloaders = {}
    test_dataloaders = {}
    validate_short_dataloaders = {}
    test_short_dataloaders = {}
    for dataset in datasets:
        train, test, validate, train_short, test_short, validate_short = loader(dataset, params['num_ctx_select'])
        train_set.extend(train)
        train_short_set.extend(train_short)
        validate_dataloader = DataLoader(EventDataset(validate), batch_size=batch_size, shuffle=True,collate_fn=collate_fn, worker_init_fn=seed_worker)
        test_dataloader = DataLoader(EventDataset(test), batch_size=batch_size, shuffle=True,collate_fn=collate_fn, worker_init_fn=seed_worker)
        validate_dataloaders[dataset] = validate_dataloader
        test_dataloaders[dataset] = test_dataloader
        if len(validate_short) == 0:
            validate_short_dataloader = None
        else:
            validate_short_dataloader = DataLoader(EventDataset(validate_short), batch_size=batch_size, shuffle=True,collate_fn=collate_fn, worker_init_fn=seed_worker)
        if len(test_short) == 0:
            test_short_dataloader = None
        else:
            test_short_dataloader = DataLoader(EventDataset(test_short), batch_size=batch_size, shuffle=True,collate_fn=collate_fn, worker_init_fn=seed_worker)
        validate_short_dataloaders[dataset] = validate_short_dataloader
        test_short_dataloaders[dataset] = test_short_dataloader
    if len(train_short_set) == 0:
        train_short_dataloader = None
    else:
        train_short_dataloader = DataLoader(EventDataset(train_short_set), batch_size=batch_size, shuffle=True,collate_fn=collate_fn, worker_init_fn=seed_worker)
    train_dataloader = DataLoader(EventDataset(train_set), batch_size=batch_size, shuffle=True,collate_fn=collate_fn, worker_init_fn=seed_worker)
    
    selector = LSTMSelector(768, params['s_hidden_dim'], params['s_mlp_dim'])
    predictor =ECIRobertaJointTask(mlp_size=params['p_mlp_dim'], roberta_type=roberta_type, datasets=datasets, pos_dim=20, 
                                    fn_activate=fn_activative, drop_rate=drop_rate, task_weights=None, n_head=params['n_head'])
    
    if CUDA:
        selector = selector.cuda()
        predictor = predictor.cuda()
    selector.zero_grad()
    predictor.zero_grad()
    print("# of parameters:", count_parameters(selector) + count_parameters(predictor))
    epoches = params['epoches'] + 5
    total_steps = len(train_dataloader) * epoches
    print("Total steps: [number of batches] x [number of epochs] =", total_steps)

    exp = EXP(selector, predictor, epoches, params['num_ctx_select'], train_dataloader, validate_dataloaders, test_dataloaders,
            train_short_dataloader, test_short_dataloaders, validate_short_dataloaders, params['s_lr'], params['p_lr'], best_path)
    F1, CM, matres_F1 = exp.train()
    exp.evaluate(is_test=True)
    print("Result: Best micro F1 of interaction: {}".format(F1))
    with open(result_file, 'a', encoding='UTF-8') as f:
        f.write("\n -------------------------------------------- \n")
        f.write("Hypeparameter: {}\n ".format(params))
        # f.write("Seed: {}\n".format(seed))
        # f.write("Drop rate: {}\n".format(drop_rate))
        # f.write("Batch size: {}\n".format(batch_size))
        # f.write("Activate function: {}\n".format(fn_activative))
        f.write("Sub: {} - Mul: {}".format(is_sub, is_mul))
        f.write("\n Best F1 MATRES: {} \n".format(matres_F1))
        for i in range(0, len(datasets)):
            f.write("{} \n".format(dataset[i]))
            f.write("F1: {} \n".format(F1[i]))
            f.write("CM: \n {} \n".format(CM[i]))
        f.write("Time: {} \n".format(datetime.datetime.now()))
    return matres_F1


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='SEED', default=0, type=int)
    parser.add_argument('--dataset', help="Name of dataset", action='append', required=True)
    parser.add_argument('--roberta_type', help="base or large", default='roberta-base', type=str)
    parser.add_argument('--best_path', help="Path for save model", type=str)
    parser.add_argument('--log_file', help="Path of log file", type=str)

    args = parser.parse_args()
    seed = args.seed
    datasets = args.dataset
    roberta_type  = args.roberta_type
    best_path = args.best_path
    best_path = [best_path+"selector.pth", best_path+"predictor.pth"]
    result_file = args.log_file

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

