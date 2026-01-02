"""Training script for ViT on occluded aircraft classification."""

import os
import random
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path BEFORE importing src modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.data.stimulus_dataset import create_dataloaders
from src.data.image_dataset import create_image_split_dataloaders
from src.models.vit_model import (
    create_vit_tiny,
    create_vit_small,
    create_vit_base,
    create_vit_large,
)
from src.models.pretrained_loader import (
    create_vit_b16_pretrained,
    create_resnet50_pretrained,
)


class Trainer:
    """Trainer for ViT classification."""

    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Set random seeds for reproducibility
        self._set_seed(self.config.get('seed', 42))

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create dataloaders
        self.dataloaders = self._create_dataloaders()

        # Create model
        self.model = self._create_model()
        self.model = self.model.to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup loss function
        self.criterion = self._create_criterion()

        # Setup logging
        self.writer = None
        if self.config['logging']['use_tensorboard']:
            log_dir = Path(self.config['logging']['log_dir'])
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.early_stop_counter = 0

        # Training history tracking
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rate': []
        }

        # Setup AMP
        self.use_amp = self.config['training'].get('use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _create_dataloaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        """Create train/val/test dataloaders."""
        dataset_config = self.config['dataset']
        dataset_type = dataset_config.get('type', 'subject_split')

        if dataset_type == 'image_split':
            # New mode: load from data/train, data/val, data/test
            print("Using image-based split mode")
            return create_image_split_dataloaders(
                train_dir=dataset_config['train_dir'],
                val_dir=dataset_config['val_dir'],
                test_dir=dataset_config.get('test_dir'),
                batch_size=dataset_config['batch_size'],
                img_size=dataset_config['image_size'],
                num_workers=dataset_config['num_workers'],
            )
        else:
            # Original mode: load by subject IDs
            print("Using subject-based split mode")
            return create_dataloaders(
                dataset_root=dataset_config['root'],
                train_subjects=dataset_config['train_subjects'],
                val_subjects=dataset_config['val_subjects'],
                test_subjects=dataset_config.get('test_subjects'),
                batch_size=dataset_config['batch_size'],
                img_size=dataset_config['image_size'],
                num_workers=dataset_config['num_workers'],
                occlusion_levels=dataset_config.get('occlusion_levels'),
            )

    def _create_model(self) -> nn.Module:
        """Create model (either pretrained from timm or custom ViT)."""
        model_config = self.config['model']
        model_type = model_config.get('type', None)
        num_classes = model_config['num_classes']

        # New: Support for pretrained models from timm
        if model_type == 'vit_b16':
            pretrained = model_config.get('pretrained', False)
            freeze_backbone = model_config.get('freeze_backbone', False)
            drop_rate = model_config.get('drop_rate', 0.0)
            drop_path_rate = model_config.get('drop_path_rate', 0.0)

            print(f"Creating ViT-B/16 (pretrained={pretrained}, freeze={freeze_backbone})")
            return create_vit_b16_pretrained(
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )

        elif model_type == 'resnet50':
            pretrained = model_config.get('pretrained', False)
            freeze_backbone = model_config.get('freeze_backbone', False)

            print(f"Creating ResNet-50 (pretrained={pretrained}, freeze={freeze_backbone})")
            return create_resnet50_pretrained(
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
            )

        # Legacy: Custom ViT models (backward compatible)
        else:
            model_size = model_config.get('size', 'base')
            image_size = self.config['dataset']['image_size']

            model_factory = {
                'tiny': create_vit_tiny,
                'small': create_vit_small,
                'base': create_vit_base,
                'large': create_vit_large,
            }

            if model_size not in model_factory:
                raise ValueError(f"Unknown model size: {model_size}")

            print(f"Creating custom ViT-{model_size}")
            return model_factory[model_size](num_classes=num_classes, image_size=image_size)

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        opt_config = self.config['training']['optimizer']
        opt_type = opt_config['type'].lower()

        if opt_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config.get('betas', [0.9, 0.999]),
            )
        elif opt_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.05),
                betas=opt_config.get('betas', [0.9, 0.999]),
            )
        elif opt_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 1e-4),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        sched_config = self.config['training']['scheduler']
        sched_type = sched_config.get('type', 'none').lower()

        if sched_type == 'cosine':
            warmup_epochs = sched_config.get('warmup_epochs', 0)
            total_epochs = self.config['training']['epochs']
            min_lr = sched_config.get('min_lr', 1e-6)

            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=min_lr,
            )
        elif sched_type == 'step':
            step_size = sched_config.get('step_size', 30)
            gamma = sched_config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        elif sched_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")

    def _create_criterion(self) -> nn.Module:
        """Create loss function."""
        loss_type = self.config['training']['loss'].lower()

        if loss_type == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {self.current_epoch}")
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config['training'].get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                if self.config['training'].get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )

                self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            if batch_idx % self.config['logging']['print_freq'] == 0:
                pbar.set_postfix({'loss': loss.item()})

        # Compute metrics
        epoch_loss = running_loss / len(self.dataloaders['train'])
        epoch_acc = accuracy_score(all_labels, all_preds)

        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels, _ in tqdm(self.dataloaders['val'], desc="Validation"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        val_loss = running_loss / len(self.dataloaders['val'])
        val_acc = accuracy_score(all_labels, all_preds)

        # Additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )

        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_config = self.config['training']['checkpoint']
        save_dir = Path(checkpoint_config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save best model
        if is_best and checkpoint_config.get('save_best', True):
            best_path = save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

        # Save last checkpoint
        if checkpoint_config.get('save_last', True):
            last_path = save_dir / 'last_checkpoint.pth'
            torch.save(checkpoint, last_path)

        # Save periodic checkpoint
        if self.current_epoch % checkpoint_config.get('save_frequency', 10) == 0:
            epoch_path = save_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
            torch.save(checkpoint, epoch_path)

    def _plot_training_history(self):
        """Plot training history and save to file."""
        import matplotlib.pyplot as plt
        import pandas as pd

        if len(self.history['epoch']) == 0:
            print("No training history to plot")
            return

        # Determine save directory
        checkpoint_config = self.config['training']['checkpoint']
        save_dir = Path(checkpoint_config['save_dir']).parent  # Go up one level from checkpoints/
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Training History - {self.config['experiment']['name']}", fontsize=16)

        epochs = self.history['epoch']

        # Plot 1: Loss
        ax = axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Accuracy
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: F1 Score
        ax = axes[1, 0]
        ax.plot(epochs, self.history['val_f1'], 'g-', label='Val F1', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Validation F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Learning Rate
        ax = axes[1, 1]
        ax.plot(epochs, self.history['learning_rate'], 'purple', label='Learning Rate', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()

        # Save figure
        plot_path = save_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {plot_path}")
        plt.close()

        # Save CSV
        df = pd.DataFrame(self.history)
        csv_path = save_dir / 'training_history.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved training history CSV to {csv_path}")

    def train(self):
        """Main training loop."""
        epochs = self.config['training']['epochs']
        early_stop_config = self.config['training']['early_stopping']

        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            # Train
            train_metrics = self.train_epoch()
            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}")

            # Validate
            val_metrics = self.validate()
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
                self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # Record history
            self.history['epoch'].append(self.current_epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Check for best model
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            # Save checkpoint
            self.save_checkpoint(is_best=is_best)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping
            if early_stop_config.get('enabled', False):
                patience = early_stop_config.get('patience', 10)
                if self.early_stop_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f}")

        # Plot training history
        self._plot_training_history()

        if self.writer is not None:
            self.writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model on occluded aircraft classification")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/vit_b16_pretrained.yaml',
        help='Path to configuration file (relative to project root)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='vit_b16',
        help='Model name (vit_b16, resnet50, etc.)'
    )
    args = parser.parse_args()

    # Resolve config path relative to project root
    config_path = project_root / args.config if not Path(args.config).is_absolute() else args.config

    trainer = Trainer(str(config_path))
    trainer.train()
