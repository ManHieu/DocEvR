import datetime
import os
from doc_exp import EXP
from utils.constant import CUDA
from models.doc_tranformer import DocTransformer
import torch
torch.manual_seed(1741)
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import random
random.seed(1741)
import numpy as np
np.random.seed(1741)
import optuna
from torch.utils.data.dataloader import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from data_loader.loader import loader
from data_loader.EventDataset import EventDataset
import gc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def collate_fn(batch):
    return tuple(zip(*batch))

def objective(trial: optuna.Trial):
    params = {
        'p_mlp_dim': trial.suggest_categorical('p_mlp_dim', [512, 768]),
        "epoches": trial.suggest_categorical("epoches", [3, 5]),
        'lr': trial.suggest_categorical("lr", [1.5e-5, 1e-5, 7e-6, 5e-6]),
        'word_drop_rate': 0.05,
        'seed': 1741
    }
    torch.manual_seed(1741)
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    drop_rate = 0.5
    fn_activative = 'relu6'
    # trial.suggest_categorical('fn_activate', ['relu', 'tanh', 'relu6', 'silu', 'hardtanh'])
    is_mul = True
    # trial.suggest_categorical('is_mul', [True, False])
    is_sub = True
    # trial.suggest_categorical('is_sub', [True, False])

    train_set = []
    validate_dataloaders = {}
    test_dataloaders = {}
    for dataset in datasets:
        train, test, validate, train_short, test_short, validate_short = loader(dataset, 7)
        train_set.extend(train + train_short)
        validate_dataloader = DataLoader(EventDataset(validate + validate_short), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
        test_dataloader = DataLoader(EventDataset(test + test_short), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
        validate_dataloaders[dataset] = validate_dataloader
        test_dataloaders[dataset] = test_dataloader
    train_dataloader = DataLoader(EventDataset(train_set), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

    print("Hyperparameter will be use in this trial: \n {}".format(params))

    predictor =DocTransformer(mlp_size=params['p_mlp_dim'], roberta_type=roberta_type, datasets=datasets, pos_dim=16, 
                                    fn_activate=fn_activative, drop_rate=drop_rate, task_weights=None)
    if CUDA:
        predictor = predictor.cuda()
    predictor.zero_grad()
    print(predictor)
    print("# of parameters:", count_parameters(predictor))
    epoches = params['epoches']
    total_steps = len(train_dataloader) * epoches
    print("Total steps: [number of batches] x [number of epochs] =", total_steps)

    exp = EXP(predictor, params['epoches'], train_dataloader, validate_dataloaders, test_dataloaders, params['lr'], best_path)
    F1, CM, matres_F1, test_f1 = exp.train()
    # test_f1 = exp.evaluate(is_test=True)
    print("Result: Best micro F1 of interaction: {}".format(F1))
    with open(result_file, 'a', encoding='UTF-8') as f:
        f.write("\n -------------------------------------------- \n")
        f.write("\nNote: {} \n".format(roberta_type))
        f.write("{}\n".format(roberta_type))
        f.write("Hypeparameter: \n{}\n ".format(params))
        f.write("Test F1: {}\n".format(test_f1))
        f.write("Seed: {}\n".format(seed))
        # f.write("Drop rate: {}\n".format(drop_rate))
        # f.write("Batch size: {}\n".format(batch_size))
        f.write("Activate function: {}\n".format(fn_activative))
        f.write("Sub: {} - Mul: {}".format(is_sub, is_mul))
        # f.write("\n Best F1 MATRES: {} \n".format(matres_F1))
        for i in range(0, len(datasets)):
            f.write("{} \n".format(dataset[i]))
            f.write("F1: {} \n".format(F1[i]))
            f.write("CM: \n {} \n".format(CM[i]))
        f.write("Time: {} \n".format(datetime.datetime.now()))
    os.rename(best_path, best_path+'.{}'.format(test_f1))

    del exp
    del predictor
    gc.collect()

    return test_f1


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='SEED', default=1741, type=int)
    parser.add_argument('--dataset', help="Name of dataset", action='append', required=True)
    parser.add_argument('--roberta_type', help="base or large", default='roberta-base', type=str)
    parser.add_argument('--best_path', help="Path for save model", type=str)
    parser.add_argument('--log_file', help="Path of log file", type=str)
    parser.add_argument('--bs', help='batch size', default=16, type=int)
    # parser.add_argument('--num_select', help='number of select sentence', default=3, type=int)

    args = parser.parse_args()
    seed = args.seed
    datasets = args.dataset
    print(datasets)
    roberta_type  = args.roberta_type
    best_path = args.best_path
    result_file = args.log_file
    batch_size = args.bs
    pre_processed_dir = "./" + "_".join(datasets) + "/"

    sampler = optuna.samplers.TPESampler(seed=1741)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=50)
    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

