import os
import os.path as osp
import sys
from shutil import copyfile
import yaml
import torch
import torch.nn as nn

from train_utils import instansiate_train_dev_test

sys.path.insert(0, osp.abspath('.'))

import argparse
from model_factory.models import NVLRformer


def _print(something):
    print(something, flush=True)
    return


def save_all(run_name, model, epoch, acc):
    torch.save({
        'epoch': epoch,
        'val_accuracy': acc,
        'model_state_dict': model.state_dict()
    }, f'./checkpoints/{run_name}_{epoch}_{acc}.pt')
    return


def load(model, model_name):
    checkpoint = torch.load(f'./checkpoints/{model_name}.pt')
    # removes 'module' from dict entries, pytorch bug #3805
    if torch.cuda.device_count() >= 1 and any(k.startswith('module.') for k in checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in
                                          checkpoint['model_state_dict'].items()}
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_accuracy']
    _print(f"Restoring model of run {model_name} at epoch {epoch} and {val_acc} validation accuracy\n")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def kwarg_list_to_device(data_obj, device):
    if device == 'cpu':
        return data_obj
    cpy = []
    for item in data_obj:
        cpy.append(item.to(device))
    return cpy


def get_accuracy(y_prob, y_true):
    y_true = y_true.squeeze(1).detach()
    y_prob = y_prob.squeeze(1).detach()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


def train_model(run_name, config, device):
    with open(config, 'r') as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    if device == 'cuda':
        device = 'cuda:0'

    train_dataloader, val_dataloader, test_dataloader = instansiate_train_dev_test(
        batch_size=int(config['batch_size']), validation_split=0.2)
    _print(f"Loaded Train Dataset at {len(train_dataloader)} batches of size {config['batch_size']}")
    _print(f"Loaded Validation Dataset at {len(val_dataloader)} batches of size {config['batch_size']}")
    model = NVLRformer(config)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['lr'], weight_decay=1e-4)

    init_epoch = 0
    ### Multi-GPU-Support ###
    if torch.cuda.device_count() > 1:
        _print(f"Let's use {torch.cuda.device_count()} GPUS")
        model = nn.DataParallel(model)

    criterion = nn.BCELoss()
    metric = get_accuracy
    best_val_acc = -1
    patience = 0
    optimizer.zero_grad()

    for epoch in range(init_epoch, config['max_epochs']):
        total_train_loss = []
        total_train_acc = []
        _print(f"Epoch: {epoch}\n")
        model.train()
        for train_batch_index, train_batch in enumerate(train_dataloader):
            data = kwarg_list_to_device(train_batch, device)
            y_real = data[-1]

            y_pred, att, _ = model(*data)
            loss = criterion(y_pred, y_real.float())
            acc = metric(y_pred, y_real)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss.append(loss.item())
            total_train_acc.append(acc)

        # End of Epoch #
        train_loss_ = sum(total_train_loss) / len(total_train_loss)
        train_acc_ = sum(total_train_acc) / len(total_train_acc)
        _print('| epoch {:3d} | loss {:5.2f} | acc {:5.2f}\n'.format(epoch, train_loss_,
                                                                     train_acc_))

        if epoch % config['validate_every_epochs'] == 0 and epoch > 0:
            _print(
                f"Validating at Epoch: {epoch}\n")
            total_val_loss = []
            total_val_acc = []
            # Turn off the train mode #
            model.eval()
            with torch.no_grad():
                for val_batch_index, val_batch in enumerate(val_dataloader):
                    data = kwarg_list_to_device(val_batch, device)
                    y_real = data[-1]
                    y_pred, att, _ = model(*data)
                    val_loss = criterion(y_pred, y_real.float())
                    val_acc = metric(y_pred, y_real)
                    total_val_loss.append(val_loss.item())
                    total_val_acc.append(val_acc)
            val_loss_ = sum(total_val_loss) / len(total_val_loss)
            val_acc_ = sum(total_val_acc) / len(total_val_acc)
            _print('| epoch {:3d} | val loss {:5.2f} | val acc {:5.2f} \n'.format(epoch,
                                                                                  val_loss_,
                                                                                  val_acc_))
            if val_acc_ > best_val_acc:
                _print("Saving Model...")
                save_all(run_name, model, epoch, val_acc_)
                if osp.exists(f'./checkpoints/{run_name}_{epoch}_{val_acc_}.pt'):
                    _print("Model saved successfully!\n")
                    patience = 0
                else:
                    _print("Failed to save Model... Aborting\n")
                    return
            else:
                patience += 1
                if patience % config['early_stopping'] == 0 and patience > 0:
                    _print(f"Training stopped at epoch: {epoch} and best validation acc: {best_val_acc}")
                    return
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='The name of the experiment', default='test_run')
    parser.add_argument('--config', type=str, help='The path to the config file', default='./model_config.yaml')
    parser.add_argument('--device', type=str, help='cpu or cuda', default='cuda')
    args = parser.parse_args()
    train_model(run_name=args.name, config=args.config, device=args.device)
