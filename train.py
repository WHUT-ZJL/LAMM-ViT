import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import time
import datetime
import gc
import json
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import MaskGuidedVisionTransformer


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MaskedFeaturesDataset(Dataset):
    """Loads pre-extracted features from HDF5 files."""

    def __init__(self, feature_files):
        self.feature_files = feature_files if isinstance(feature_files, list) else [feature_files]
        self.sample_info = []
        for file in self.feature_files:
            if os.path.exists(file):
                with h5py.File(file, 'r') as f:
                    self.sample_info.extend([(file, name) for name in f if name.startswith('sample_')])

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        file, name = self.sample_info[idx]
        with h5py.File(file, 'r') as f:
            sample = f[name]
            image = torch.tensor(sample['image'][:], dtype=torch.float32)
            masks = torch.tensor(sample['region_masks'][:], dtype=torch.float32)
            label = torch.tensor(sample.attrs['label'], dtype=torch.long)
            return image, masks, label


def create_dataloaders(feature_dir, batch_size, datasets_to_use, num_workers, rank, world_size):
    """Create data loaders for training and validation."""
    dataloaders = {}

    train_datasets, val_datasets = [], []
    for name in datasets_to_use:
        for type_ in ["real", "fake"]:
            path = os.path.join(feature_dir, type_, name)
            if os.path.isdir(path):
                train_file = os.path.join(path, "train_features.h5")
                val_file = os.path.join(path, "val_features.h5")
                if os.path.exists(train_file): train_datasets.append(MaskedFeaturesDataset(train_file))
                if os.path.exists(val_file): val_datasets.append(MaskedFeaturesDataset(val_file))

    if not train_datasets: raise ValueError("No training datasets found!")

    train_set = ConcatDataset(train_datasets)
    val_set = ConcatDataset(val_datasets) if val_datasets else None

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
                                       shuffle=True) if world_size > 1 else None
    dataloaders['train'] = DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None),
                                      sampler=train_sampler, num_workers=num_workers, pin_memory=True)

    if val_set:
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank,
                                         shuffle=False) if world_size > 1 else None
        dataloaders['val'] = DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
                                        pin_memory=True)

    return dataloaders


class MaskDiversityLoss(nn.Module):
    """Encourages diversity in attention strategies across samples in a batch."""

    def __init__(self, diversity_weight=0.2):
        super().__init__()
        self.weight = diversity_weight

    def forward(self, mask_weights_list):
        if not mask_weights_list: return torch.tensor(0.0, device=mask_weights_list[
            0].device if mask_weights_list else 'cpu')
        total_loss = 0.0
        for weights in mask_weights_list:
            B = weights.shape[0]
            if B <= 1: continue
            norm_weights = F.normalize(weights, p=2, dim=1)
            similarity = torch.matmul(norm_weights, norm_weights.t())
            # Penalize high similarity (encourage diversity)
            loss = (similarity - torch.eye(B, device=weights.device)).pow(2).mean()
            total_loss += loss
        return (total_loss / len(mask_weights_list)) * self.weight


class CombinedLoss(nn.Module):
    """Combines Cross-Entropy loss with Mask Diversity loss."""

    def __init__(self, diversity_weight=0.2):
        super().__init__()
        self.main_criterion = nn.CrossEntropyLoss()
        self.diversity_criterion = MaskDiversityLoss(diversity_weight)

    def forward(self, outputs, labels):
        main_loss = self.main_criterion(outputs['logits'], labels)
        diversity_loss = self.diversity_criterion(outputs['mask_weights'])
        return {'total': main_loss + diversity_loss, 'main': main_loss, 'diversity': diversity_loss}


def train_epoch(model, dataloader, optimizer, criterion, device, scaler, accumulation_steps, rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", disable=(rank != 0))
    for i, (images, masks, labels) in enumerate(pbar):
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(images, masks)
            losses = criterion(outputs, labels)
            loss = losses['total'] / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += losses['total'].item()
        if rank == 0:
            pbar.set_postfix(loss=losses['total'].item())

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, rank):
    """Evaluate the model."""
    model.eval()
    total_loss, all_labels, all_probs = 0, [], []

    pbar = tqdm(dataloader, desc="Evaluating", disable=(rank != 0))
    with torch.no_grad():
        for images, masks, labels in pbar:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            outputs = model(images, masks)
            total_loss += criterion(outputs, labels)['total'].item()

            probs = F.softmax(outputs['logits'], dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(all_labels, np.array(all_probs) > 0.5) * 100
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    return total_loss / len(dataloader), acc, auc


def setup_ddp(rank, world_size):
    """Initialize DDP."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP."""
    destroy_process_group()


def ddp_main(rank, world_size, config):
    """Main DDP training function."""
    if world_size > 1: setup_ddp(rank, world_size)
    set_seed(42 + rank)

    device = torch.device(f'cuda:{rank}')

    dataloaders = create_dataloaders(config['feature_dir'], config['batch_size'], config['datasets'], config['workers'],
                                     rank, world_size)

    model = MaskGuidedVisionTransformer(**config['model_params']).to(device)
    if world_size > 1: model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    criterion = CombinedLoss(config['diversity_weight']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, min_lr=config['min_lr'])
    scaler = torch.cuda.amp.GradScaler()

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(config['epochs']):
        if world_size > 1: dataloaders['train'].sampler.set_epoch(epoch)

        train_loss = train_epoch(model, dataloaders['train'], optimizer, criterion, device, scaler,
                                 config['accumulation_steps'], rank)

        if 'val' in dataloaders:
            val_loss, val_acc, val_auc = evaluate(model, dataloaders['val'], criterion, device, rank)

            if rank == 0:
                print(
                    f"Epoch {epoch + 1}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}")
                scheduler.step(val_auc)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    model_to_save = model.module if world_size > 1 else model
                    torch.save(model_to_save.state_dict(), os.path.join(config['output_dir'], 'best_model.pth'))
                    print(f"New best model saved with Val AUC: {best_val_auc:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= config['patience']:
                    print("Early stopping triggered.")
                    break

    if world_size > 1: cleanup_ddp()


if __name__ == "__main__":
    config = {
        'feature_dir': './features',
        'output_dir': './output_model',
        'datasets': ['IMDB_WIKI', 'StyleGAN3', 'StableDiffusion1.5'],
        'batch_size': 32, 'workers': 16,
        'model_params': {'dim': 768, 'num_layers': 12, 'num_heads': 12, 'mlp_dim': 3072},
        'epochs': 100, 'lr': 1e-4, 'min_lr': 1e-6, 'weight_decay': 0.05,
        'patience': 10, 'accumulation_steps': 4, 'diversity_weight': 0.2
    }
    os.makedirs(config['output_dir'], exist_ok=True)

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(ddp_main, args=(world_size, config), nprocs=world_size, join=True)
    else:
        ddp_main(0, 1, config)