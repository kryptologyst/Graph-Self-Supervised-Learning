"""Training utilities for graph self-supervised learning."""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.device import get_device
from ..models.ssl_models import NodeMaskingSSL, ContrastiveSSL, GraphCLSSL, MultiTaskSSL


class Trainer:
    """Trainer for graph self-supervised learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_wandb: bool = False,
        project_name: str = "graph-ssl"
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize wandb if requested
        if self.use_wandb:
            import wandb
            wandb.init(project=project_name)
            wandb.watch(self.model)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(
        self, 
        data: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> float:
        """Train for one epoch.
        
        Args:
            data: Graph data object.
            loss_fn: Loss function (if None, model should have compute_loss method).
            
        Returns:
            Average training loss.
        """
        self.model.train()
        data = data.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        if isinstance(self.model, MultiTaskSSL):
            outputs = self.model(data)
            loss = self.model.compute_loss(outputs)
        elif isinstance(self.model, (NodeMaskingSSL, ContrastiveSSL, GraphCLSSL)):
            if isinstance(self.model, NodeMaskingSSL):
                embeddings, reconstructed, targets = self.model(data)
                loss = self.model.compute_loss(reconstructed, targets, loss_fn)
            elif isinstance(self.model, ContrastiveSSL):
                embeddings, pos_scores, neg_scores = self.model(data)
                loss = self.model.compute_loss(pos_scores, neg_scores)
            elif isinstance(self.model, GraphCLSSL):
                orig_embeddings, aug_embeddings = self.model(data)
                loss = self.model.compute_loss(orig_embeddings, aug_embeddings)
        else:
            # Generic forward pass
            outputs = self.model(data)
            if loss_fn is not None:
                loss = loss_fn(outputs, data)
            else:
                raise ValueError("Loss function must be provided for generic models")
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return loss.item()
    
    def validate(
        self, 
        data: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> float:
        """Validate the model.
        
        Args:
            data: Graph data object.
            loss_fn: Loss function.
            
        Returns:
            Validation loss.
        """
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            if isinstance(self.model, MultiTaskSSL):
                outputs = self.model(data)
                loss = self.model.compute_loss(outputs)
            elif isinstance(self.model, (NodeMaskingSSL, ContrastiveSSL, GraphCLSSL)):
                if isinstance(self.model, NodeMaskingSSL):
                    embeddings, reconstructed, targets = self.model(data)
                    loss = self.model.compute_loss(reconstructed, targets, loss_fn)
                elif isinstance(self.model, ContrastiveSSL):
                    embeddings, pos_scores, neg_scores = self.model(data)
                    loss = self.model.compute_loss(pos_scores, neg_scores)
                elif isinstance(self.model, GraphCLSSL):
                    orig_embeddings, aug_embeddings = self.model(data)
                    loss = self.model.compute_loss(orig_embeddings, aug_embeddings)
            else:
                outputs = self.model(data)
                if loss_fn is not None:
                    loss = loss_fn(outputs, data)
                else:
                    raise ValueError("Loss function must be provided for generic models")
        
        return loss.item()
    
    def train(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        num_epochs: int = 100,
        loss_fn: Optional[nn.Module] = None,
        save_best: bool = True,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_data: Training data.
            val_data: Validation data.
            num_epochs: Number of training epochs.
            loss_fn: Loss function.
            save_best: Whether to save the best model.
            verbose: Whether to print progress.
            
        Returns:
            Training history.
        """
        if verbose:
            print(f"Training on {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_data, loss_fn)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if val_data is not None:
                val_loss = self.validate(val_data, loss_fn)
                self.val_losses.append(val_loss)
                
                # Save best model
                if save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
            
            # Logging
            if self.use_wandb:
                import wandb
                log_dict = {"epoch": epoch, "train_loss": train_loss}
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                wandb.log(log_dict)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                print(msg)
        
        training_time = time.time() - start_time
        
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            if val_data is not None and save_best:
                print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "training_time": training_time
        }
    
    def load_best_model(self):
        """Load the best model state."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        else:
            print("No best model state found")


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            model: Model to potentially save weights from.
            
        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer for the model.
    
    Args:
        model: Model to optimize.
        optimizer_name: Name of the optimizer.
        lr: Learning rate.
        weight_decay: Weight decay.
        **kwargs: Additional optimizer arguments.
        
    Returns:
        Optimizer instance.
    """
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "step",
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule.
        scheduler_name: Name of the scheduler.
        **kwargs: Additional scheduler arguments.
        
    Returns:
        Scheduler instance or None.
    """
    if scheduler_name.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name.lower() == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name.lower() == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
