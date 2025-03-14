import time
import numpy as np

import torch
import torch.nn.functional as F

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from torch import tensor
from torch.optim import Adam, SGD
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader as DenseLoader

from tqdm import tqdm



device = torch.device('cuda')

def cross_validation_with_val_set(idataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None, dataset_name='abc'):
    if idataset.__class__.__name__ == 'iRunDataset':
        folds = 1
    
    val_losses, val_metrics, metrics, durations = [], [], [], []
    train_acc = []
            
    for fold, (train_dataset, test_dataset, val_dataset) in enumerate(idataset.load_datasets()): 
        
        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
            
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

        if dataset_name in ['PCQM4Mv2', 'Peptides-struct', 'ZINC']:
            best_metric = 999.9
        else:
            best_metric = 0.0
        
        torch.cuda.synchronize()
        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):            
            start_time = time.perf_counter()
            train_loss = train(model, optimizer, train_loader, dataset_name)
                
            end_time = time.perf_counter()
            total_time = end_time - start_time
            scheduler.step()
            
            val_losses.append(eval_loss(model, val_loader, dataset_name))

            if dataset_name in ['PCQM4Mv2', 'Peptides-struct', 'ZINC']:
                metric_value = eval_mae(model, test_loader)
                metric_name = 'test_MAE'
            elif dataset_name == 'ogbg-molpcba':
                metric_value = eval_ap(model, test_loader)
                metric_name = 'test_AP'
            elif dataset_name == 'ogbg-molhiv':
                metric_value = eval_rocauc(model, test_loader)
                metric_name = 'test_ROC-AUC'
            elif dataset_name == 'Peptides-func':
                metric_value = eval_ap(model, test_loader, dataset_name)
                metric_name = 'test_AP'
            else:
                metric_value = eval_acc(model, test_loader)
                metric_name = 'test_acc' 
            if dataset_name in ['PCQM4Mv2', 'Peptides-struct', 'ZINC']:
                if best_metric >= metric_value:
                    best_metric = metric_value 
            else:
                if best_metric <= metric_value:
                    best_metric = metric_value 
                 
            metrics.append(metric_value)
                
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_losses[-1]:.4f}',
                metric_name: f'{metric_value:.4f}',
                'best': f'{best_metric:.4f}',
                'time': f'{total_time:.4f}',
            }

            print(eval_info)
            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
                    
                    

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch.backends,
                     'mps') and torch.backends.mps.is_available():
            torch.mps.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
         
 
  
    loss, metric, duration = tensor(val_losses), tensor(metrics), tensor(durations) 
    loss, metric = loss.view(folds, epochs), metric.view(folds, epochs)
    
    if dataset_name in ['Peptides-struct']:
        min, argmin = metric.min(dim=1) 
        metric = metric[torch.arange(folds, dtype=torch.long), argmin] 
    else:
        max, argmax = metric.max(dim=1)
        metric = metric[torch.arange(folds, dtype=torch.long), argmax] 
        

    loss_mean = loss.mean().item()
    metric_mean = metric.mean().item()
    metric_std = metric.std().item()
    duration_mean = duration.mean().item()
    
    
    if dataset_name in ['Peptides-struct']:
        print(f'Val Loss: {loss_mean:.4f}, Test MAE: {metric_mean:.4f} '
              f'± {metric_std:.4f}, Duration: {duration_mean:.4f}') 
    elif dataset_name == 'Peptides-func':
        print(f'Val Loss: {loss_mean:.4f}, Test AP: {metric_mean:.4f} '
              f'± {metric_std:.4f}, Duration: {duration_mean:.4f}')
    else:
        print(f'Val Loss: {loss_mean:.4f}, Test Accuracy: {metric_mean:.4f} '
              f'± {metric_std:.4f}, Duration: {duration_mean:.4f}')

    return loss_mean, metric_mean, metric_std, duration_mean


def k_fold(dataset, folds, dataset_name='abc'):
    if dataset_name in ['Peptides-struct']:
        kf = KFold(n_splits=folds, shuffle=True, random_state=1993)
        indices = list(range(len(dataset)))
        train_indices, test_indices = [], []
        
        for train_idx, test_idx in kf.split(indices):
            train_indices.append(torch.tensor(train_idx, dtype=torch.long))
            test_indices.append(torch.tensor(test_idx, dtype=torch.long))
        
        val_indices = [test_indices[i - 1] for i in range(folds)]
        return train_indices, test_indices, val_indices
    
    if dataset_name in ['Peptides-func']:
        mskf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=1993)
        y = dataset.data.y.cpu().numpy()
        train_indices, test_indices = [ ], [ ]
        
        for train_idx, test_idx in mskf.split(np.zeros(len(y)), y):
            train_indices.append(torch.from_numpy(train_idx).to(torch.long))
            test_indices.append(torch.from_numpy(test_idx).to(torch.long))
            
        val_indices = [test_indices[i - 1] for i in range(folds)]
        
        return train_indices, test_indices, val_indices
    
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1993)
        y = dataset.data.y.cpu().numpy()
        train_indices, test_indices = [ ], [ ]
        
        for _, test_idx in skf.split(torch.zeros(len(dataset)), y):
            test_indices.append(torch.from_numpy(test_idx).to(torch.long))

        val_indices = [test_indices[i - 1] for i in range(folds)]

        train_indices = []
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


def train(model, optimizer, loader, dataset_name='abc'):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        if dataset_name in ['Peptides-struct']:
            loss = F.l1_loss(out.squeeze(), data.y)
        elif dataset_name in ['Peptides-func']:
            loss = F.binary_cross_entropy_with_logits(out, data.y.float())
        else: 
            loss = F.nll_loss(out, data.y) 
 
        h0_loss = F.mse_loss(model.get_h0(data.x), -1*model.odeblock.odefunc.get_diffusion()) 
        loss = 10*loss +  h0_loss
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
        model.bm.update(model.getNFE())
        model.resetNFE()
    return total_loss / len(loader.dataset) 


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad(): 
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)
 
def eval_ap(model, loader, dataset_name='abc'):
    model.eval()
    
    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            if dataset_name == 'Peptides-func':
                probs = torch.sigmoid(pred)
            else:
                probs = torch.exp(pred)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    ap_list = []
    for i in range(all_labels.shape[1]):
        is_labeled = ~np.isnan(all_labels[:, i])
        if np.sum(is_labeled) == 0:
            continue
        ap = average_precision_score(
            all_labels[is_labeled, i],
            all_preds[is_labeled, i]
        )
        ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            "No positively labeled data available. Cannot compute Average Precision."
        )
    return np.mean(ap_list)


def eval_rocauc(model, loader):
    model.eval()
 
    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            probs = torch.sigmoid(pred)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    rocauc_list = [] 
    for i in range(all_labels.shape[1]):
        if np.sum(all_labels[:, i] == 1) > 0 and np.sum(all_labels[:, i] == 0) > 0:
            is_labeled = ~np.isnan(all_labels[:, i])
            rocauc_list.append(
                roc_auc_score(all_labels[is_labeled, i], all_preds[is_labeled, i]))
        
    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')
    
    return np.mean(rocauc_list)


def eval_loss(model, loader, dataset_name='abc'):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            if dataset_name in ['Peptides-struct']: 
                loss += 10*F.l1_loss(out.squeeze(), data.y)
            elif dataset_name in ['Peptides-func']: 
                loss += 10*F.binary_cross_entropy_with_logits(out, data.y.float())
            else: 
                loss += 10*F.nll_loss(out, data.y)
                

    return loss / len(loader.dataset)


def eval_mae(model, loader):
    model.eval()

    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).squeeze()
        all_preds.append(pred.cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return np.mean(np.abs(all_preds - all_labels))


@torch.no_grad()
def inference_run(model, loader, bf16):
    model.eval()
    for data in loader:
        data = data.to(device)
        if bf16:
            data.x = data.x.to(torch.bfloat16)
        model(data)
        