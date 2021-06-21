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
        for item in batch:
            print(item)
        break
    for batch in short_dataloader:
        print("==================== Short Batch ====================")
        for item in batch:
            print(item)
        break