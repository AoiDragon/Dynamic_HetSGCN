import torch
from param_parser import parameter_parser
from train_and_test import Trainer
from torch_geometric.loader import DataLoader
from generate_dataset import AvalonDataset

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    config = parameter_parser()
    dataset = AvalonDataset('./dataset')

    s = config.dataset_size
    train_dataset = dataset[:int(0.8 * s) * 25]
    val_dataset = dataset[int(0.8 * s) * 25:int(0.9 * s) * 25]
    test_dataset = dataset[int(0.9 * s) * 25:s * 25]

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size * 25, shuffle=False, drop_last=False,
                              num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 25, shuffle=False, drop_last=True,
                            num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 25, shuffle=False, drop_last=False,
                             num_workers=12)

    trainer = Trainer(config, train_loader, val_loader, test_loader)
    trainer.train_and_val(train_loader, val_loader)
    print("Finish training and validating")
    trainer.test(test_loader)
