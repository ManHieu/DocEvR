import datetime
import os
from exp import EXP
from utils.constant import CUDA
from models.predictor_model import ECIRobertaJointTask
from models.selector_model import LSTMSelector
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
        's_hidden_dim': 512, 
        's_mlp_dim': 512, 
        'p_mlp_dim': 512, 
        'epoches':  trial.suggest_categorical('eps', [0]), 
        'warming_epoch': trial.suggest_categorical('warming_epoch', [10, 20, 30, 50]), 
        'num_ctx_select': trial.suggest_categorical('num_ctx_select', [2]), 
        's_lr': trial.suggest_categorical('s_lr', [5e-5]), 
        'b_lr': trial.suggest_categorical('b_lr', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]), 
        'm_lr': trial.suggest_categorical('m_lr', [1e-5, 3e-5, 5e-5, 7e-5]), 
        'b_lr_decay_rate': 0.5, 
        'word_drop_rate': 0.05, 
        'task_reward': 'logit', 
        'perfomance_reward_weight': 0.7, 
        'ctx_sim_reward_weight': 0.003, 
        'knowledge_reward_weight': 0.0, 
        'fn_activate': 'tanh', 
        'seed': 1741
        }
    torch.manual_seed(1741)
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    drop_rate = 0.5
    fn_activative = params['fn_activate']
    is_mul = True
    # trial.suggest_categorical('is_mul', [True, False])
    is_sub = True
    # trial.suggest_categorical('is_sub', [True, False])
    num_select = params['num_ctx_select']

    # train_set = []
    # train_short_set = []
    # validate_dataloaders = []
    test_dataloaders = {}
    # validate_short_dataloaders = []
    test_short_dataloaders = {}
    train, test, validate, train_short, test_short, validate_short = loader(datasets[0], params['num_ctx_select']+1, sentence_encoder=model_type, lang='en')
    train_dataloader = DataLoader(EventDataset(train), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    if len(train_short) == 0:
        train_short_dataloader = None
    else:
        train_short_dataloader = DataLoader(EventDataset(train_short), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    validate_dataloader = DataLoader(EventDataset(validate), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    if len(validate_short) == 0:
        validate_short_dataloader = None
    else:
        validate_short_dataloader = DataLoader(EventDataset(validate_short), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

    langs = ['da', 'es', 'tr', 'ur']
    for la in langs:
        train, test, validate, train_short, test_short, validate_short = loader(datasets[0], params['num_ctx_select']+1, sentence_encoder=model_type, lang=la)
        test_dataloader = DataLoader(EventDataset(test), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
        if len(test_short) == 0:
            test_short_dataloader = None
        else:
            test_short_dataloader = DataLoader(EventDataset(test_short), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
        test_dataloaders[la] = test_dataloader
        test_short_dataloaders[la] = test_short_dataloader

    print("Hyperparameter will be use in this trial: \n {}".format(params))

    selector = LSTMSelector(768, params['s_hidden_dim'], params['s_mlp_dim'])
    predictor =ECIRobertaJointTask(mlp_size=params['p_mlp_dim'], roberta_type=model_type, datasets=datasets, pos_dim=16, 
                                    fn_activate=fn_activative, drop_rate=drop_rate, task_weights=None)
    # print(predictor)
    if CUDA:
        selector = selector.cuda()
        predictor = predictor.cuda()
    selector.zero_grad()
    predictor.zero_grad()
    print("# of parameters:", count_parameters(selector) + count_parameters(predictor))
    epoches = params['epoches'] + 5
    total_steps = len(train_dataloader) * epoches
    print("Total steps: [number of batches] x [number of epochs] =", total_steps)
    print(f"Hyperparams: \n{params}")
    exp = EXP(selector, predictor, epoches, params['num_ctx_select'], train_dataloader, validate_dataloader, test_dataloaders,
            train_short_dataloader, test_short_dataloaders, validate_short_dataloader, 
            params['s_lr'], params['b_lr'], params['m_lr'], params['b_lr_decay_rate'],  params['epoches'], params['warming_epoch'],
            best_path, word_drop_rate=params['word_drop_rate'], reward=[params['task_reward']], perfomance_reward_weight=params['perfomance_reward_weight'],
            ctx_sim_reward_weight=params['ctx_sim_reward_weight'], kg_reward_weight=params['knowledge_reward_weight'])
    best_f1, best_result = exp.train()
    # test_f1 = exp.evaluate(is_test=True)
    print("Result: Best micro F1 of interaction: {}".format(best_f1[-1]))
    with open(result_file, 'a', encoding='UTF-8') as f:
        f.write("\n -------------------------------------------- \n")
        # f.write("\nNote: use lstm in predictor \n")
        # f.write("{}\n".format(lang))
        f.write("{}\n".format(model_type))
        f.write("Hypeparameter: \n{}\n ".format(params))
        f.write("Test F1: {}\n".format(best_f1))
        f.write("Best results: {}\n".format(best_result))
        f.write("Seed: {}\n".format(seed))
        # f.write("Drop rate: {}\n".format(drop_rate))
        # f.write("Batch size: {}\n".format(batch_size))
        f.write("Activate function: {}\n".format(fn_activative))
        f.write("Sub: {} - Mul: {}".format(is_sub, is_mul))
        # f.write("\n Best F1 MATRES: {} \n".format(matres_F1))
        f.write("Time: {} \n".format(datetime.datetime.now()))

    del exp
    del selector
    del predictor
    gc.collect()

    return best_f1[-1]


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='SEED', default=1741, type=int)
    parser.add_argument('--dataset', help="Name of dataset", action='append', required=True)
    # parser.add_argument('--lang', help="sub dataset", required=True)
    parser.add_argument('--model_type', help="pretrained model type", default='roberta-base', type=str)
    parser.add_argument('--best_path', help="Path for save model", type=str)
    parser.add_argument('--log_file', help="Path of log file", type=str)
    parser.add_argument('--bs', help='batch size', default=16, type=int)
    # parser.add_argument('--num_select', help='number of select sentence', default=3, type=int)

    args = parser.parse_args()
    seed = args.seed
    datasets = args.dataset
    # lang = args.lang
    # print(f"{datasets} - {lang}")
    model_type  = args.model_type
    best_path = args.best_path
    if not os.path.exists(best_path):
        os.mkdir(best_path)
    best_path = [best_path+"selector.pth", best_path+"predictor.pth"]
    result_file = args.log_file
    batch_size = args.bs
    pre_processed_dir = "./" + "_".join(datasets) + "/"

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=50)
    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

