from torch._C import dtype
from .utils.tools import *
import torch


def make_selector_input(x_target, y_target, x_ctx, y_ctx, x_sent_id, y_sent_id):
    bs = len(x_target)
    x_ctx_augm = []
    y_ctx_augm = []
    x_target_len = []
    y_target_len = []
    x_ctx_len = []
    y_ctx_len = []
    for i in range(bs):
        x_ctx_augm.append(create_augmented_ctx(x_target[i], x_ctx[i], x_sent_id[i]))
        y_ctx_augm.append(create_augmented_ctx(y_target[i], y_ctx[i], y_sent_id[i]))
        x_target_len.append(len(x_target[i]))
        y_target_len.append(len(y_target[i]))
        x_ctx_lens = [len(sent) for sent in x_ctx[i]]
        y_ctx_lens = [len(sent) for sent in y_ctx[i]]
        x_ctx_len.append(x_ctx_lens)
        y_ctx_len.append(y_ctx_lens)

    x_target = torch.tensor(padding_matrix(x_target), dtype=torch.long)
    y_target = torch.tensor(padding_matrix(y_target), dtype=torch.long)
    x_ctx_augm = torch.tensor(x_ctx_augm, dtype=torch.long)
    y_ctx_augm = torch.tensor(y_ctx_augm, dtype=torch.long)
    x_target_len = torch.tensor(x_target_len, dtype=torch.long)
    y_target_len = torch.tensor(y_target_len, dtype=torch.long)
    x_ctx_len = torch.tensor(x_ctx_len, dtype=torch.long)
    y_ctx_len = torch.tensor(y_ctx_len, dtype=torch.long)

    return x_target, y_target, x_ctx_augm, y_ctx_augm, x_target_len, y_target_len, x_ctx_len, y_ctx_len

def create_augmented_ctx(target, ctx, target_sent_id):
    augmented_ctx = []
    for i in range(len(ctx)):
        ctx_sent = ctx[i]
        if i < target_sent_id:
            augmented = ctx_sent +  target[1:]
        else:
            augmented = target + ctx[1:]
        
        # augmented = padding(augmented)
        augmented_ctx.append(augmented)
    return augmented_ctx

if __name__ == '__main__':
    pass


