"""
Training infrastructure for multimodal genre classification.

This module implements the Trainer class which handles:
- Training loop with validation
- Checkpointing and model saving
- Early stopping
- Learning rate scheduling
- Mixed precision training (AMP)
- TensorBoard logging
- Gradient clipping and accumulation

Classes:
    EarlyStopping: Early stopping callback
    Trainer: Main training class

Functions:
    create_optimizer: Factory function for optimizers
    create_scheduler: Factory function for learning rate schedulers
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from tqdm import tqdm
import time
import json


class EarlyStopping:
    """
    Early stopping callback to stop training when validation metric stops improving.

    Attributes:
        patience (int): Number of epochs to wait before stopping
        min_delta (float): Minimum change to qualify as improvement
        mode (str): 'min' or 'max' (minimize or maximize metric)
        counter (int): Current count of epochs without improvement
        best_score (Optional[float]): Best score observed
        early_stop (bool): Flag indicating whether to stop

    Example:
        >>> early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='max')
        >>> for epoch in range(100):
        ...     val_metric = train_one_epoch()
        ...     early_stopping(val_metric)
        ...     if early_stopping.early_stop:
        ...         print("Early stopping triggered")
        ...         break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        """
        Initialize early stopping callback.

        Args:
            patience (int, optional): Number of epochs with no improvement
                after which training will be stopped. Defaults to 10.
            min_delta (float, optional): Minimum change in monitored quantity
                to qualify as improvement. Defaults to 0.0.
            mode (str, optional): One of 'min' or 'max'. In 'min' mode, training
                stops when metric stops decreasing; in 'max' mode, stops when
                metric stops increasing. Defaults to 'max'.
            verbose (bool, optional): Print messages. Defaults to True.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric: float) -> None:
        """
        Check if training should stop.

        Args:
            metric (float): Current validation metric value
        """
        if self.best_score is None:
            # First epoch
            self.best_score = metric
            if self.verbose:
                print(f"Early stopping: Initial metric = {metric:.4f}")
        else:
            # Check for improvement
            if self.mode == 'max':
                is_improved = metric > (self.best_score + self.min_delta)
            else:  # mode == 'min'
                is_improved = metric < (self.best_score - self.min_delta)

            if is_improved:
                # Improvement detected
                self.best_score = metric
                self.counter = 0
                if self.verbose:
                    print(f"Early stopping: Metric improved to {metric:.4f}")
            else:
                # No improvement
                self.counter += 1
                if self.verbose:
                    print(
                        f"Early stopping: No improvement for {self.counter}/{self.patience} epochs "
                        f"(current: {metric:.4f}, best: {self.best_score:.4f})"
                    )

                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(
                            f"Early stopping triggered! No improvement for {self.patience} epochs. "
                            f"Best metric: {self.best_score:.4f}"
                        )

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class Trainer:
    """
    Trainer class for training multimodal genre classification models.

    Handles complete training pipeline including:
    - Training and validation loops
    - Checkpointing best and last models
    - Early stopping
    - Learning rate scheduling
    - Mixed precision training
    - Gradient clipping and accumulation
    - TensorBoard logging

    Attributes:
        model (nn.Module): Model to train
        optimizer (Optimizer): Optimizer
        loss_fn (nn.Module): Loss function
        device (torch.device): Device to train on
        config (Dict[str, Any]): Training configuration

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=loss_fn,
        ...     device=device,
        ...     config=config
        ... )
        >>> history = trainer.train(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     num_epochs=50
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        scheduler: Optional[_LRScheduler] = None,
        metric_fn: Optional[Callable] = None,
        checkpoint_dir: Optional[Path] = None,
        resume_from: Optional[str] = None
    ):
        """
        Initialize Trainer.

        Args:
            model (nn.Module): Model to train
            optimizer (Optimizer): Optimizer for training
            loss_fn (nn.Module): Loss function
            device (torch.device): Device to train on (cuda/cpu/mps)
            config (Dict[str, Any]): Training configuration dictionary
            scheduler (Optional[_LRScheduler], optional): Learning rate scheduler.
                Defaults to None.
            metric_fn (Optional[Callable], optional): Function to compute validation metric.
                Should take (predictions, targets) and return float. Defaults to None.
            checkpoint_dir (Optional[Path], optional): Directory to save checkpoints.
                Defaults to None (uses config['checkpoint_dir']).
            resume_from (Optional[str], optional): Path to checkpoint to resume from.
                Defaults to None.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.metric_fn = metric_fn

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        early_stop_config = config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.001),
            mode=early_stop_config.get('mode', 'max'),
            verbose=early_stop_config.get('verbose', True)
        ) if early_stop_config.get('enabled', True) else None

        # Mixed precision training
        self.use_amp = config.get('use_amp', False) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.grad_accumulation_steps = config.get('grad_accumulation_steps', 1)

        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', None)

        # TensorBoard logging
        self.use_tensorboard = config.get('use_tensorboard', True)
        if self.use_tensorboard:
            log_dir = Path(config.get('log_dir', 'runs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metric': [],
            'learning_rates': []
        }

        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader (DataLoader): Training data loader
            epoch (int): Current epoch number

        Returns:
            Dict[str, float]: Dictionary with training metrics (loss, etc.)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        # Progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{self.config.get('num_epochs', 50)} [Train]",
            leave=False
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self._forward_pass(batch)
                    loss = self.loss_fn(outputs, batch['labels'])
                    loss = loss / self.grad_accumulation_steps
            else:
                outputs = self._forward_pass(batch)
                loss = self.loss_fn(outputs, batch['labels'])
                loss = loss / self.grad_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Update metrics
            total_loss += loss.item() * self.grad_accumulation_steps

            # Update progress bar
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

            # Log to TensorBoard
            if self.writer and (batch_idx % self.config.get('log_interval', 10) == 0):
                self.writer.add_scalar(
                    'train/batch_loss',
                    loss.item() * self.grad_accumulation_steps,
                    self.global_step
                )

        avg_loss = total_loss / num_batches

        return {'loss': avg_loss}

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader (DataLoader): Validation data loader
            epoch (int): Current epoch number

        Returns:
            Dict[str, float]: Dictionary with validation metrics (loss, metric, etc.)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        # Progress bar
        pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{self.config.get('num_epochs', 50)} [Val]",
            leave=False
        )

        for batch in pbar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            outputs = self._forward_pass(batch)
            loss = self.loss_fn(outputs, batch['labels'])

            # Update metrics
            total_loss += loss.item()

            # Collect predictions and targets for metric computation
            predictions = torch.sigmoid(outputs)  # Convert logits to probabilities
            all_predictions.append(predictions.cpu())
            all_targets.append(batch['labels'].cpu())

        avg_loss = total_loss / len(val_loader)

        # Compute metric if metric function provided
        if self.metric_fn is not None:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            metric = self.metric_fn(all_predictions, all_targets)
        else:
            metric = -avg_loss  # Use negative loss as default metric (higher is better)

        return {
            'loss': avg_loss,
            'metric': metric
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ) -> Dict[str, list]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (Optional[int], optional): Number of epochs to train.
                If None, uses config['num_epochs']. Defaults to None.

        Returns:
            Dict[str, list]: Training history with losses and metrics
        """
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 50)

        print(f"\n{'='*80}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.grad_accumulation_steps}")
        print(f"Max gradient norm: {self.max_grad_norm}")
        print(f"{'='*80}\n")

        start_time = time.time()

        for epoch in range(self.current_epoch + 1, self.current_epoch + num_epochs + 1):
            epoch_start_time = time.time()

            # Train one epoch
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs metric value
                    self.scheduler.step(val_metrics['metric'])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_metric'].append(val_metrics['metric'])
            self.history['learning_rates'].append(current_lr)

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('epoch/val_metric', val_metrics['metric'], epoch)
                self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)

            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            print(
                f"Epoch {epoch}/{self.current_epoch + num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Metric: {val_metrics['metric']:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Save checkpoint
            is_best = self._is_best_metric(val_metrics['metric'])
            self.save_checkpoint(
                epoch=epoch,
                is_best=is_best,
                metric=val_metrics['metric']
            )

            # Early stopping check
            if self.early_stopping is not None:
                self.early_stopping(val_metrics['metric'])
                if self.early_stopping.early_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

            self.current_epoch = epoch

        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation metric: {self.best_metric:.4f}")
        print(f"{'='*80}\n")

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()

        return self.history

    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform forward pass based on model type.

        Args:
            batch (Dict[str, torch.Tensor]): Batch dictionary

        Returns:
            torch.Tensor: Model outputs (logits)
        """
        # Determine model type and forward accordingly
        # This handles text-only, vision-only, and multimodal models

        if 'text' in batch and 'image' in batch:
            # Multimodal model
            if 'attention_mask' in batch:
                # BERT-based text model
                outputs = self.model(
                    text_input={'input_ids': batch['text'], 'attention_mask': batch['attention_mask']},
                    image_input=batch['image']
                )
            else:
                # LSTM-based text model
                outputs = self.model(
                    text_input=batch['text'],
                    image_input=batch['image']
                )
        elif 'text' in batch:
            # Text-only model
            if 'attention_mask' in batch:
                outputs = self.model(batch['text'], batch['attention_mask'])
            else:
                outputs = self.model(batch['text'])
        elif 'image' in batch:
            # Vision-only model
            outputs = self.model(batch['image'])
        else:
            raise ValueError("Batch must contain 'text' and/or 'image' keys")

        return outputs

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move batch tensors to device.

        Args:
            batch (Dict[str, torch.Tensor]): Batch dictionary

        Returns:
            Dict[str, torch.Tensor]: Batch with tensors on device
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def _is_best_metric(self, metric: float) -> bool:
        """
        Check if current metric is best so far.

        Args:
            metric (float): Current metric value

        Returns:
            bool: True if best metric
        """
        mode = self.early_stopping.mode if self.early_stopping else 'max'

        if self.best_metric is None:
            self.best_metric = metric
            return True
        else:
            if mode == 'max':
                is_best = metric > self.best_metric
            else:
                is_best = metric < self.best_metric

            if is_best:
                self.best_metric = metric
            return is_best

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        metric: Optional[float] = None
    ) -> None:
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch
            is_best (bool, optional): Whether this is the best model so far.
                Defaults to False.
            metric (Optional[float], optional): Current metric value. Defaults to None.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric,
            'history': self.history,
            'config': self.config
        }

        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last.pth'
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint (metric: {metric:.4f})")

        # Save epoch checkpoint (optional)
        if self.config.get('save_every_epoch', False):
            epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load checkpoint and resume training.

        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.history = checkpoint['history']

        print(f"Resumed from epoch {self.current_epoch}, best metric: {self.best_metric:.4f}")


# ============================================================================
# Optimizer and Scheduler Factory Functions
# ============================================================================

def create_optimizer(
    model: nn.Module,
    optimizer_config: Dict[str, Any]
) -> Optimizer:
    """
    Factory function to create optimizers.

    Args:
        model (nn.Module): Model to optimize
        optimizer_config (Dict[str, Any]): Optimizer configuration

    Returns:
        Optimizer: Initialized optimizer

    Example:
        >>> config = {
        ...     'type': 'adamw',
        ...     'lr': 0.001,
        ...     'weight_decay': 0.01,
        ...     'betas': (0.9, 0.999)
        ... }
        >>> optimizer = create_optimizer(model, config)
    """
    optimizer_type = optimizer_config.get('type', 'adamw').lower()
    lr = optimizer_config.get('lr', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 0.01)

    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=optimizer_config.get('nesterov', False)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    scheduler_config: Dict[str, Any]
) -> Optional[_LRScheduler]:
    """
    Factory function to create learning rate schedulers.

    Args:
        optimizer (Optimizer): Optimizer to schedule
        scheduler_config (Dict[str, Any]): Scheduler configuration

    Returns:
        Optional[_LRScheduler]: Initialized scheduler or None

    Example:
        >>> config = {
        ...     'type': 'cosine',
        ...     'T_max': 50,
        ...     'eta_min': 1e-6
        ... }
        >>> scheduler = create_scheduler(optimizer, config)
    """
    if not scheduler_config or not scheduler_config.get('enabled', True):
        return None

    scheduler_type = scheduler_config.get('type', 'cosine').lower()

    if scheduler_type == 'cosine' or scheduler_type == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 50),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step' or scheduler_type == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau' or scheduler_type == 'reducelronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'max'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 5),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'exponential' or scheduler_type == 'exponentiallr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.95)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


# Example usage
if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("Use this module to train your models with comprehensive training infrastructure.")
