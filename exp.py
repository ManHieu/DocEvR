from os import path
import numpy
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import tqdm
from models.predictor_model import ECIRobertaJointTask
from models.selector_model import LSTMSelector, SelectorModel
import time
from utils.tools import *
import torch
import torch.optim as optim


class EXP(object):
    def __init__(self, selector, predictor, num_epoches, num_ctx_select,
                train_dataloader, validate_dataloaders, test_dataloaders, 
                train_short_dataloader, test_short_dataloaders, validate_short_dataloaders,
                s_lr, p_lr, best_path) -> None:
        super().__init__()
        self.selector = selector
        self.predictor = predictor
        if isinstance(self.selector, SelectorModel):
            self.is_finetune_selector = True
        if isinstance(self.selector, LSTMSelector):
            self.is_finetune_selector = False

        self.num_epoches = num_epoches
        self.num_ctx_select = num_ctx_select

        self.train_dataloader = train_dataloader
        self.test_datatloaders = list(test_dataloaders.values())
        self.validate_dataloaders = list(validate_dataloaders.values())
        self.train_short_dataloader = train_short_dataloader
        self.test_short_datatloaders = list(test_short_dataloaders.values())
        self.validate_short_dataloaders = list(validate_short_dataloaders.values())
        self.datasets = list(test_dataloaders.keys())

        self.s_lr = s_lr
        self.p_lr = p_lr
        self.selector_optim = optim.AdamW(self.selector.parameters(), lr=s_lr, amsgrad=True)
        self.predictor_optim = optim.AdamW(self.predictor.parameters(), lr=self.p_lr, amsgrad=True)
        
        self.best_micro_f1 = [0.0]*len(self.test_datatloaders)
        self.sum_f1 = 0.0
        self.best_matres = 0.0
        self.best_cm = [None]*len(self.test_datatloaders)
        self.best_path_selector = best_path[0]
        self.best_path_predictor = best_path[1]

    def train(self):
        start_time = time.time()
        for i in range(self.num_epoches):
            print("============== Epoch {} / {} ==============".format(i+1, self.num_epoches))
            t0 = time.time()
            self.selector.train()
            self.selector.zero_grad()
            self.selector_loss = 0.0

            self.predictor.train()
            self.predictor.zero_grad()
            self.predictor_loss = 0.0
            for step, batch in tqdm.tqdm(enumerate(self.train_dataloader), desc="Training process", total=len(self.train_dataloader)):
                x_sent_id, y_sent_id, x_sent, y_sent, x_sent_emb, y_sent_emb, x_position, y_position, x_sent_pos, y_sent_pos, x_ctx, y_ctx,  \
                x_ctx_len, y_ctx_len, x_ctx_augm, y_ctx_augm, x_ctx_augm_emb, y_ctx_augm_emb, x_ctx_pos, y_ctx_pos, flag, xy = batch
                
                self.selector_optim.zero_grad()
                self.predictor_optim.zero_grad()

                if self.is_finetune_selector == False:
                    target_emb, ctx_emb, target_len, ctx_len = make_selector_input()
                    x_ctx_selected, x_dist, x_log_prob = self.selector()


