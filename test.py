if __name__ == '__main__':
    import numpy as np
    from data_loader.EventDataset import EventDataset
    from torch.utils.data.dataloader import DataLoader
    import torch
    import random
    from data_loader.loader import loader
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train, test, validate, train_short, test_short, validate_short = loader("MATRES", 3)
    dataloader = DataLoader(EventDataset(train), batch_size=12, shuffle=True,collate_fn=collate_fn, worker_init_fn=seed_worker)
    short_dataloader = DataLoader(EventDataset(train_short), batch_size=12, shuffle=True,collate_fn=collate_fn, worker_init_fn=seed_worker)
    for batch in dataloader:
        print("==================== Batch ====================")
        x_sent_id, y_sent_id, x_sent, y_sent, x_sent_len, y_sent_len, x_sent_emb, y_sent_emb, x_position, y_position, x_sent_pos, y_sent_pos, \
        x_ctx, y_ctx, x_ctx_len, y_ctx_len, x_ctx_augm, y_ctx_augm, x_ctx_augm_emb, y_ctx_augm_emb, x_ctx_pos, y_ctx_pos, flag, xy = batch
        print(x_ctx_augm_emb)
        print(x_ctx_augm_emb.size())
        break
    for batch in short_dataloader:
        print("==================== Short Batch ====================")
        for item in batch:
            print(item)
        break