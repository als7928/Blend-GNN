import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import tensor
from torch.optim import Adam

from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

from torcheval.metrics import BinaryAUROC
from torch.utils.data.dataloader import default_collate
device = torch.device('cuda')

def my_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    return default_collate(batch)


def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None, dataset_name='abc'):

    val_losses, accs, durations = [], [], []
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        elif model.__class__.__name__ == "AugRD":
            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=my_collate)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=my_collate)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=my_collate)
        else:
            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

        # if torch.cuda.is_available():
        torch.cuda.synchronize()

        t_start = time.perf_counter()
        # epoch_5 = 0
        # epoch_loss = 9999
        # best_val_loss = 99999 # initialize as a sufficiently large number for each fold before training
        for epoch in range(1, epochs + 1):
            start_time = time.perf_counter()
            if dataset_name == 'ogbg-molhiv':
                train_loss, auc_roc = train_molhiv(model, optimizer, train_loader)
            else:
                train_loss = train(model, optimizer, train_loader) 
            end_time = time.perf_counter()
            total_time = end_time - start_time
            scheduler.step()

            val_losses.append(eval_loss(model, val_loader))
            accs.append(eval_acc(model, test_loader))
            if dataset_name == 'ogbg-molhiv':
                eval_info = {
                    'fold': fold,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_losses[-1],
                    'test_acc': accs[-1],
                    'auc_roc': auc_roc,
                    'time': total_time,
                }
            else:
                eval_info = {
                    'fold': fold,
                    'epoch': epoch,
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_losses[-1]:.4f}',
                    'test_acc': f'{accs[-1]*100:.2f}',
                    'time': f'{total_time:.4f}',
                }
            print(eval_info)
            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            
            # if float(eval_info['val_loss']) < best_val_loss: # check if this validation loss is the lowest
            #     best_val_loss = float(eval_info['val_loss'])  # set this loss as the best validation loss
            #     best_epoch = epoch # set this epoch as the best epoch
            #     # best_weights = model.state_dict()

            # if (epoch - best_epoch) > patient: # if no updates after several epochs, stop at this fold
            #     print(f'Early stopping at epoch {epoch} after the best epoch {best_epoch}.')
            #     for i in range(epoch+1, epochs+1): # pad to fix the length of array for each fold
            #         val_losses.append(99999) # append a sufficiently large number
            #         accs.append(-1) # append a sufficiently small number
            #     break # early stop
            # if eval_info['val_loss'] < epoch_loss:
            #     print('gg')
            #     epoch_loss = eval_info['val_loss']
            #     best_weights = model.state_dict()

            # if epoch_5 > 20:
            #     model.load_state_dict(best_weights)
            #     epoch_5 = 0
            #     epoch_loss = 9999
            # epoch_5+=1

        if torch.cuda.is_available(): 
            torch.cuda.synchronize()
        elif hasattr(torch.backends,
                     'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc, argacc = acc.max(dim=1)
    # acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {acc_mean:.3f} '
          f'± {acc_std:.3f}, Duration: {duration_mean:.3f}')

    return loss_mean, acc_mean, acc_std, duration_mean


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset._data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
    
    return train_indices, test_indices, val_indices


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device) #DataBatch(edge_index=[2, 5884], x=[2667, 7], y=[150], batch=[2667], ptr=[151])

        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        if model.__class__.__name__ == "AugRD":
            if model.odeblock.nreg > 0:  # add regularisation - slower for small data, but faster and better performance for large data
                reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
                regularization_coeffs = model.regularization_coeffs

                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                )
                loss = loss + reg_loss
            model.fm.update(model.getNFE())
            model.resetNFE()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        if model.__class__.__name__ == "AugRD":
            model.bm.update(model.getNFE())
            model.resetNFE()
    return total_loss / len(loader.dataset)

def train_molhiv(model, optimizer, loader):
    auc_roc = BinaryAUROC().to(device)
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        # print(data) #DataBatch(edge_index=[2, 225448], x=[104606, 9], y=[4096, 1], num_nodes=104606, batch=[104606], ptr=[4097])
        data.x = data.x.float()
        out = model(data) # [graphs, 2]
        pred = (out > 0.5).float().squeeze()
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        auc_roc.update(input=out.squeeze(), target=data.y)
    auc_roc = auc_roc.compute().detach().cpu()

    return total_loss / len(loader.dataset), auc_roc


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader):
    model.eval()
    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


@torch.no_grad()
def inference_run(model, loader, bf16):
    model.eval()
    for data in loader:
        data = data.to(device)
        if bf16:
            data.x = data.x.to(torch.bfloat16)
        model(data)
