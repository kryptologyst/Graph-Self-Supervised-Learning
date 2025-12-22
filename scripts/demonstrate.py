#!/usr/bin/env python3
"""Demonstration script for the modernized Graph Self-Supervised Learning project."""

import torch
import numpy as np
from pathlib import Path
import json
import time

# Import our modernized modules
from src.models.encoders import GCNEncoder, AttributeDecoder, ContrastiveDecoder
from src.models.ssl_models import NodeMaskingSSL, ContrastiveSSL, GraphCLSSL
from src.data.datasets import load_dataset, create_train_val_test_splits, preprocess_graph
from src.train.trainer import Trainer, create_optimizer
from src.eval.evaluator import GraphSSLEvaluator, create_evaluation_report
from src.utils.device import get_device, set_seed


def demonstrate_node_masking():
    """Demonstrate node masking SSL."""
    print("\n" + "="*60)
    print("🔍 DEMONSTRATING NODE MASKING SSL")
    print("="*60)
    
    # Load synthetic data for quick demonstration
    print("📊 Loading synthetic dataset...")
    data, dataset_info = load_dataset("synthetic_sbm_1000")
    data = preprocess_graph(data)
    data = create_train_val_test_splits(data, random_seed=42)
    
    print(f"Dataset: {dataset_info}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_node_features}")
    
    # Create model
    print("\n🏗️  Creating Node Masking SSL model...")
    encoder = GCNEncoder(
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=64,
        num_layers=2,
        dropout=0.5,
        use_batch_norm=True
    )
    decoder = AttributeDecoder(
        in_channels=64,
        out_channels=data.num_node_features,
        hidden_channels=64
    )
    model = NodeMaskingSSL(encoder, decoder, mask_ratio=0.3)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n🚀 Training model...")
    device = get_device()
    model.to(device)
    data = data.to(device)
    
    optimizer = create_optimizer(model, "adam", lr=0.01)
    trainer = Trainer(model, optimizer, device=device)
    
    start_time = time.time()
    history = trainer.train(data, num_epochs=20, verbose=True)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    evaluator = GraphSSLEvaluator(device=device)
    
    # Node classification
    nc_metrics = evaluator.evaluate_node_classification(model, data)
    print("Node Classification Results:")
    for metric, value in nc_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Link prediction
    lp_metrics = evaluator.evaluate_link_prediction(model, data)
    print("\nLink Prediction Results:")
    for metric, value in lp_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Clustering
    cluster_metrics = evaluator.evaluate_clustering(model, data)
    print("\nClustering Results:")
    for metric, value in cluster_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    return model, data, history


def demonstrate_contrastive_learning():
    """Demonstrate contrastive SSL."""
    print("\n" + "="*60)
    print("🔗 DEMONSTRATING CONTRASTIVE LEARNING SSL")
    print("="*60)
    
    # Load data
    data, dataset_info = load_dataset("synthetic_sbm_1000")
    data = preprocess_graph(data)
    data = create_train_val_test_splits(data, random_seed=42)
    
    print(f"Dataset: {dataset_info}")
    
    # Create model
    encoder = GCNEncoder(
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=64,
        num_layers=2,
        dropout=0.5
    )
    contrastive_decoder = ContrastiveDecoder(64, hidden_channels=32)
    model = ContrastiveSSL(encoder, contrastive_decoder, temperature=0.1)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    device = get_device()
    model.to(device)
    data = data.to(device)
    
    optimizer = create_optimizer(model, "adam", lr=0.01)
    trainer = Trainer(model, optimizer, device=device)
    
    print("\n🚀 Training contrastive model...")
    history = trainer.train(data, num_epochs=20, verbose=True)
    
    # Evaluate
    evaluator = GraphSSLEvaluator(device=device)
    nc_metrics = evaluator.evaluate_node_classification(model, data)
    
    print("\nContrastive Learning Results:")
    for metric, value in nc_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    return model, data


def demonstrate_graphcl():
    """Demonstrate GraphCL SSL."""
    print("\n" + "="*60)
    print("🎨 DEMONSTRATING GRAPHCL SSL")
    print("="*60)
    
    # Load data
    data, dataset_info = load_dataset("synthetic_sbm_1000")
    data = preprocess_graph(data)
    data = create_train_val_test_splits(data, random_seed=42)
    
    print(f"Dataset: {dataset_info}")
    
    # Create model
    encoder = GCNEncoder(
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=64,
        num_layers=2,
        dropout=0.5
    )
    contrastive_decoder = ContrastiveDecoder(64, hidden_channels=32)
    model = GraphCLSSL(encoder, contrastive_decoder, temperature=0.1, aug_ratio=0.1)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    device = get_device()
    model.to(device)
    data = data.to(device)
    
    optimizer = create_optimizer(model, "adam", lr=0.01)
    trainer = Trainer(model, optimizer, device=device)
    
    print("\n🚀 Training GraphCL model...")
    history = trainer.train(data, num_epochs=20, verbose=True)
    
    # Evaluate
    evaluator = GraphSSLEvaluator(device=device)
    nc_metrics = evaluator.evaluate_node_classification(model, data)
    
    print("\nGraphCL Results:")
    for metric, value in nc_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    return model, data


def compare_methods():
    """Compare different SSL methods."""
    print("\n" + "="*60)
    print("📊 COMPARING SSL METHODS")
    print("="*60)
    
    # Load data once
    data, dataset_info = load_dataset("synthetic_sbm_1000")
    data = preprocess_graph(data)
    data = create_train_val_test_splits(data, random_seed=42)
    
    device = get_device()
    data = data.to(device)
    
    methods = {
        "Node Masking": NodeMaskingSSL,
        "Contrastive": ContrastiveSSL,
        "GraphCL": GraphCLSSL
    }
    
    results = {}
    
    for method_name, model_class in methods.items():
        print(f"\n🔬 Testing {method_name}...")
        
        # Create model
        encoder = GCNEncoder(
            in_channels=data.num_node_features,
            hidden_channels=64,
            out_channels=64,
            num_layers=2,
            dropout=0.5
        )
        
        if method_name == "Node Masking":
            decoder = AttributeDecoder(64, data.num_node_features)
            model = model_class(encoder, decoder, mask_ratio=0.3)
        else:
            contrastive_decoder = ContrastiveDecoder(64, hidden_channels=32)
            if method_name == "Contrastive":
                model = model_class(encoder, contrastive_decoder, temperature=0.1)
            else:  # GraphCL
                model = model_class(encoder, contrastive_decoder, temperature=0.1, aug_ratio=0.1)
        
        # Train
        model.to(device)
        optimizer = create_optimizer(model, "adam", lr=0.01)
        trainer = Trainer(model, optimizer, device=device)
        
        start_time = time.time()
        trainer.train(data, num_epochs=15, verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate
        evaluator = GraphSSLEvaluator(device=device)
        nc_metrics = evaluator.evaluate_node_classification(model, data)
        
        results[method_name] = {
            "accuracy": nc_metrics.get("accuracy", 0.0),
            "f1_macro": nc_metrics.get("f1_macro", 0.0),
            "training_time": training_time
        }
    
    # Print comparison
    print("\n📈 METHOD COMPARISON RESULTS:")
    print("-" * 50)
    print(f"{'Method':<15} {'Accuracy':<10} {'F1-Macro':<10} {'Time(s)':<10}")
    print("-" * 50)
    
    for method, metrics in results.items():
        print(f"{method:<15} {metrics['accuracy']:<10.4f} {metrics['f1_macro']:<10.4f} {metrics['training_time']:<10.2f}")
    
    return results


def main():
    """Main demonstration function."""
    print("🕸️  GRAPH SELF-SUPERVISED LEARNING DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates the modernized Graph SSL project")
    print("with multiple self-supervised learning methods.")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device info
    device = get_device()
    print(f"\n🖥️  Using device: {device}")
    
    try:
        # Demonstrate different methods
        print("\n🎯 Running demonstrations...")
        
        # Node Masking
        model1, data1, history1 = demonstrate_node_masking()
        
        # Contrastive Learning
        model2, data2 = demonstrate_contrastive_learning()
        
        # GraphCL
        model3, data3 = demonstrate_graphcl()
        
        # Compare methods
        comparison_results = compare_methods()
        
        # Save results
        output_dir = Path("outputs/demonstration")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "comparison_results.json", "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_dir}")
        
        # Summary
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✅ Modern GCN architecture with batch normalization")
        print("✅ Multiple SSL methods (Node Masking, Contrastive, GraphCL)")
        print("✅ Comprehensive evaluation metrics")
        print("✅ Device-agnostic training")
        print("✅ Reproducible experiments")
        print("✅ Clean, modular code structure")
        
        print("\nNext Steps:")
        print("1. Run the interactive demo: streamlit run demo/app.py")
        print("2. Train on real datasets: python -m src.train.main --dataset cora")
        print("3. Explore the codebase in src/ directory")
        print("4. Run tests: pytest tests/")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
