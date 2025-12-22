"""Main training script for graph self-supervised learning."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.encoders import GCNEncoder, AttributeDecoder, ContrastiveDecoder
from src.models.ssl_models import NodeMaskingSSL, ContrastiveSSL, GraphCLSSL, MultiTaskSSL
from src.data.datasets import load_dataset, create_train_val_test_splits, preprocess_graph
from src.train.trainer import Trainer, create_optimizer, create_scheduler, EarlyStopping
from src.eval.evaluator import GraphSSLEvaluator, create_evaluation_report
from src.utils.device import get_device, set_seed


def create_model(config: Dict[str, Any], num_features: int) -> nn.Module:
    """Create SSL model based on configuration.
    
    Args:
        config: Model configuration.
        num_features: Number of input features.
        
    Returns:
        SSL model instance.
    """
    model_config = config["model"]
    
    # Create encoder
    encoder_config = model_config["encoder"]
    encoder = GCNEncoder(
        in_channels=num_features,
        hidden_channels=encoder_config["hidden_channels"],
        out_channels=encoder_config["out_channels"],
        num_layers=encoder_config["num_layers"],
        dropout=encoder_config["dropout"],
        use_batch_norm=encoder_config["use_batch_norm"],
        use_residual=encoder_config["use_residual"]
    )
    
    # Create decoder
    decoder_config = model_config["decoder"]
    if decoder_config["type"] == "attribute":
        decoder = AttributeDecoder(
            in_channels=encoder_config["out_channels"],
            out_channels=num_features,
            hidden_channels=decoder_config["hidden_channels"],
            num_layers=decoder_config["num_layers"],
            dropout=decoder_config["dropout"]
        )
    elif decoder_config["type"] == "contrastive":
        decoder = ContrastiveDecoder(
            in_channels=encoder_config["out_channels"],
            hidden_channels=decoder_config["hidden_channels"],
            dropout=decoder_config["dropout"]
        )
    else:
        raise ValueError(f"Unknown decoder type: {decoder_config['type']}")
    
    # Create SSL model
    model_type = model_config["type"]
    
    if model_type == "node_masking":
        model = NodeMaskingSSL(
            encoder=encoder,
            decoder=decoder,
            mask_ratio=model_config["mask_ratio"],
            mask_value=model_config["mask_value"]
        )
    elif model_type == "contrastive":
        contrastive_decoder = ContrastiveDecoder(
            in_channels=encoder_config["out_channels"],
            hidden_channels=decoder_config["hidden_channels"],
            dropout=decoder_config["dropout"]
        )
        model = ContrastiveSSL(
            encoder=encoder,
            decoder=contrastive_decoder,
            temperature=model_config["temperature"]
        )
    elif model_type == "graphcl":
        contrastive_decoder = ContrastiveDecoder(
            in_channels=encoder_config["out_channels"],
            hidden_channels=decoder_config["hidden_channels"],
            dropout=decoder_config["dropout"]
        )
        model = GraphCLSSL(
            encoder=encoder,
            decoder=contrastive_decoder,
            temperature=model_config["temperature"],
            aug_ratio=model_config["aug_ratio"]
        )
    elif model_type == "multi_task":
        attribute_decoder = AttributeDecoder(
            in_channels=encoder_config["out_channels"],
            out_channels=num_features,
            hidden_channels=decoder_config["hidden_channels"],
            num_layers=decoder_config["num_layers"],
            dropout=decoder_config["dropout"]
        )
        contrastive_decoder = ContrastiveDecoder(
            in_channels=encoder_config["out_channels"],
            hidden_channels=decoder_config["hidden_channels"],
            dropout=decoder_config["dropout"]
        )
        model = MultiTaskSSL(
            encoder=encoder,
            attribute_decoder=attribute_decoder,
            contrastive_decoder=contrastive_decoder,
            mask_ratio=model_config["mask_ratio"],
            temperature=model_config["temperature"],
            task_weights=model_config["task_weights"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Graph Self-Supervised Learning")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--dataset", type=str, help="Override dataset name")
    parser.add_argument("--model", type=str, help="Override model type")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, help="Override device")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--output", type=str, help="Override output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.dataset:
        config.data.dataset_name = args.dataset
    if args.model:
        config.model.type = args.model
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.system.device = args.device
    if args.seed:
        config.system.seed = args.seed
    if args.output:
        config.output.save_dir = args.output
    
    # Set random seed
    set_seed(config.system.seed)
    
    # Get device
    if config.system.device == "auto":
        device = get_device()
    else:
        device = torch.device(config.system.device)
    
    print(f"Using device: {device}")
    print(f"Configuration: {config.model.type} on {config.data.dataset_name}")
    
    # Create output directory
    output_dir = Path(config.output.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    data, dataset_info = load_dataset(
        config.data.dataset_name,
        root=config.data.root,
        **config.data.get("synthetic", {})
    )
    
    print(f"Dataset: {dataset_info}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_node_features}")
    
    # Preprocess data
    data = preprocess_graph(
        data,
        add_self_loops=config.data.add_self_loops,
        make_undirected=config.data.make_undirected,
        normalize_features=config.data.normalize_features
    )
    
    # Create train/val/test splits
    data = create_train_val_test_splits(
        data,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.system.seed
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config, data.num_node_features)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.2f} MB")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        optimizer_name=config.training.optimizer,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=config.training.scheduler,
        **config.training.scheduler_params
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        use_wandb=config.training.use_wandb,
        project_name=config.training.wandb_project
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(**config.training.early_stopping)
    
    # Training loop with early stopping
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config.training.num_epochs):
        # Training
        train_loss = trainer.train_epoch(data)
        
        # Validation
        val_loss = trainer.validate(data)
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
        # Update scheduler
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        
        # Logging
        if (epoch + 1) % config.training.log_interval == 0:
            print(f"Epoch {epoch+1:3d}/{config.training.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config.output.save_model:
                torch.save(model.state_dict(), output_dir / "best_model.pt")
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Load best model for evaluation
    if config.output.save_model and (output_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    
    # Evaluation
    print("Evaluating model...")
    evaluator = GraphSSLEvaluator(device=device)
    evaluation_report = create_evaluation_report(model, data, evaluator)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print("=" * 50)
    for task, metrics in evaluation_report.items():
        print(f"\n{task.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save results
    if config.output.save_predictions:
        import json
        with open(output_dir / "evaluation_results.json", "w") as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for task, metrics in evaluation_report.items():
                json_results[task] = {k: float(v) for k, v in metrics.items()}
            json.dump(json_results, f, indent=2)
    
    # Save embeddings
    if config.output.save_embeddings:
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'encoder'):
                embeddings = model.encoder(data.x, data.edge_index)
            else:
                embeddings = model(data.x, data.edge_index)
            
            torch.save({
                'embeddings': embeddings.cpu(),
                'labels': data.y.cpu() if hasattr(data, 'y') else None,
                'train_mask': data.train_mask.cpu() if hasattr(data, 'train_mask') else None,
                'val_mask': data.val_mask.cpu() if hasattr(data, 'val_mask') else None,
                'test_mask': data.test_mask.cpu() if hasattr(data, 'test_mask') else None,
            }, output_dir / "embeddings.pt")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
