"""Evaluation utilities for graph self-supervised learning."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    normalized_mutual_info_score, adjusted_rand_score
)
from torchmetrics import Accuracy, F1Score, AUROC
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from ..utils.device import get_device


class GraphSSLEvaluator:
    """Evaluator for graph self-supervised learning models."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        
        # Initialize metrics
        self.metrics = {
            "accuracy": Accuracy(task="multiclass"),
            "f1_micro": F1Score(task="multiclass", average="micro"),
            "f1_macro": F1Score(task="multiclass", average="macro"),
            "auroc": AUROC(task="multiclass", num_classes=None)
        }
    
    def evaluate_node_classification(
        self,
        model: nn.Module,
        data: torch.Tensor,
        task: str = "node_masking"
    ) -> Dict[str, float]:
        """Evaluate model on node classification task.
        
        Args:
            model: Trained SSL model.
            data: Graph data with train/val/test masks.
            task: SSL task type.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            # Get embeddings
            if hasattr(model, 'encoder'):
                embeddings = model.encoder(data.x, data.edge_index)
            else:
                # Assume model outputs embeddings directly
                embeddings = model(data.x, data.edge_index)
            
            # Train a simple classifier on embeddings
            classifier = self._train_classifier(embeddings, data)
            
            # Evaluate on test set
            test_metrics = self._evaluate_classifier(
                classifier, embeddings, data, split="test"
            )
        
        return test_metrics
    
    def evaluate_link_prediction(
        self,
        model: nn.Module,
        data: torch.Tensor,
        num_neg_samples: int = 1000
    ) -> Dict[str, float]:
        """Evaluate model on link prediction task.
        
        Args:
            model: Trained SSL model.
            data: Graph data.
            num_neg_samples: Number of negative samples for evaluation.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            # Get embeddings
            if hasattr(model, 'encoder'):
                embeddings = model.encoder(data.x, data.edge_index)
            else:
                embeddings = model(data.x, data.edge_index)
            
            # Create positive and negative pairs
            pos_pairs = data.edge_index.t()
            neg_pairs = self._sample_negative_pairs(
                data.num_nodes, num_neg_samples
            )
            
            # Compute similarities
            pos_scores = self._compute_similarity_scores(embeddings, pos_pairs)
            neg_scores = self._compute_similarity_scores(embeddings, neg_pairs)
            
            # Compute metrics
            metrics = self._compute_link_prediction_metrics(pos_scores, neg_scores)
        
        return metrics
    
    def evaluate_clustering(
        self,
        model: nn.Module,
        data: torch.Tensor,
        n_clusters: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model on clustering task.
        
        Args:
            model: Trained SSL model.
            data: Graph data with ground truth labels.
            n_clusters: Number of clusters (if None, use number of classes).
            
        Returns:
            Dictionary of evaluation metrics.
        """
        from sklearn.cluster import KMeans
        
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            # Get embeddings
            if hasattr(model, 'encoder'):
                embeddings = model.encoder(data.x, data.edge_index)
            else:
                embeddings = model(data.x, data.edge_index)
            
            # Perform clustering
            if n_clusters is None:
                n_clusters = len(torch.unique(data.y))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings.cpu().numpy())
            
            # Compute metrics
            true_labels = data.y.cpu().numpy()
            
            metrics = {
                "nmi": normalized_mutual_info_score(true_labels, cluster_labels),
                "ari": adjusted_rand_score(true_labels, cluster_labels),
                "silhouette": self._compute_silhouette_score(embeddings.cpu().numpy(), cluster_labels)
            }
        
        return metrics
    
    def evaluate_reconstruction_quality(
        self,
        model: nn.Module,
        data: torch.Tensor,
        mask_ratio: float = 0.3
    ) -> Dict[str, float]:
        """Evaluate reconstruction quality for node masking task.
        
        Args:
            model: Trained SSL model.
            data: Graph data.
            mask_ratio: Ratio of nodes to mask for evaluation.
            
        Returns:
            Dictionary of reconstruction metrics.
        """
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            # Create masked data
            from ..utils.data_utils import mask_node_features
            masked_x, mask_indices = mask_node_features(
                data.x, mask_ratio=mask_ratio
            )
            
            # Forward pass
            if isinstance(model, NodeMaskingSSL):
                embeddings, reconstructed, targets = model(data)
            else:
                # Generic reconstruction
                embeddings = model.encoder(masked_x, data.edge_index)
                reconstructed = model.decoder(embeddings[mask_indices])
                targets = data.x[mask_indices]
            
            # Compute reconstruction metrics
            mse = torch.nn.functional.mse_loss(reconstructed, targets)
            mae = torch.nn.functional.l1_loss(reconstructed, targets)
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                reconstructed, targets, dim=1
            ).mean()
            
            metrics = {
                "mse": mse.item(),
                "mae": mae.item(),
                "cosine_similarity": cos_sim.item()
            }
        
        return metrics
    
    def _train_classifier(
        self,
        embeddings: torch.Tensor,
        data: torch.Tensor,
        hidden_dim: int = 64
    ) -> nn.Module:
        """Train a simple MLP classifier on embeddings."""
        classifier = nn.Sequential(
            nn.Linear(embeddings.size(1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, len(torch.unique(data.y)))
        ).to(self.device)
        
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        classifier.train()
        for epoch in range(100):
            optimizer.zero_grad()
            
            train_embeddings = embeddings[data.train_mask]
            train_labels = data.y[data.train_mask]
            
            logits = classifier(train_embeddings)
            loss = criterion(logits, train_labels)
            
            loss.backward()
            optimizer.step()
        
        return classifier
    
    def _evaluate_classifier(
        self,
        classifier: nn.Module,
        embeddings: torch.Tensor,
        data: torch.Tensor,
        split: str = "test"
    ) -> Dict[str, float]:
        """Evaluate classifier on specified split."""
        classifier.eval()
        
        if split == "test":
            mask = data.test_mask
        elif split == "val":
            mask = data.val_mask
        else:
            raise ValueError(f"Unknown split: {split}")
        
        with torch.no_grad():
            test_embeddings = embeddings[mask]
            test_labels = data.y[mask]
            
            logits = classifier(test_embeddings)
            predictions = torch.argmax(logits, dim=1)
            
            # Compute metrics
            accuracy = accuracy_score(test_labels.cpu(), predictions.cpu())
            f1_micro = f1_score(test_labels.cpu(), predictions.cpu(), average="micro")
            f1_macro = f1_score(test_labels.cpu(), predictions.cpu(), average="macro")
            
            # AUROC (one-vs-rest)
            try:
                auroc = roc_auc_score(
                    test_labels.cpu().numpy(),
                    torch.softmax(logits, dim=1).cpu().numpy(),
                    multi_class="ovr",
                    average="macro"
                )
            except ValueError:
                auroc = 0.0  # Handle case with single class
        
        return {
            "accuracy": accuracy,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "auroc": auroc
        }
    
    def _sample_negative_pairs(
        self,
        num_nodes: int,
        num_samples: int
    ) -> torch.Tensor:
        """Sample negative node pairs."""
        src_nodes = torch.randint(0, num_nodes, (num_samples,))
        dst_nodes = torch.randint(0, num_nodes, (num_samples,))
        
        # Ensure src != dst
        same_node_mask = src_nodes == dst_nodes
        dst_nodes[same_node_mask] = (dst_nodes[same_node_mask] + 1) % num_nodes
        
        return torch.stack([src_nodes, dst_nodes], dim=1)
    
    def _compute_similarity_scores(
        self,
        embeddings: torch.Tensor,
        pairs: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity scores for node pairs."""
        src_embeddings = embeddings[pairs[:, 0]]
        dst_embeddings = embeddings[pairs[:, 1]]
        
        # Cosine similarity
        scores = torch.nn.functional.cosine_similarity(
            src_embeddings, dst_embeddings, dim=1
        )
        
        return scores
    
    def _compute_link_prediction_metrics(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> Dict[str, float]:
        """Compute link prediction metrics."""
        # Combine scores and labels
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([
            torch.ones(pos_scores.size(0)),
            torch.zeros(neg_scores.size(0))
        ])
        
        # AUROC
        try:
            auroc = roc_auc_score(all_labels.cpu().numpy(), all_scores.cpu().numpy())
        except ValueError:
            auroc = 0.0
        
        # Average Precision
        from sklearn.metrics import average_precision_score
        try:
            ap = average_precision_score(all_labels.cpu().numpy(), all_scores.cpu().numpy())
        except ValueError:
            ap = 0.0
        
        return {
            "auroc": auroc,
            "average_precision": ap
        }
    
    def _compute_silhouette_score(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray
    ) -> float:
        """Compute silhouette score."""
        from sklearn.metrics import silhouette_score
        try:
            return silhouette_score(embeddings, cluster_labels)
        except ValueError:
            return 0.0  # Handle case with single cluster


def create_evaluation_report(
    model: nn.Module,
    data: torch.Tensor,
    evaluator: Optional[GraphSSLEvaluator] = None
) -> Dict[str, Dict[str, float]]:
    """Create comprehensive evaluation report.
    
    Args:
        model: Trained SSL model.
        data: Graph data.
        evaluator: Evaluator instance.
        
    Returns:
        Dictionary containing evaluation results for different tasks.
    """
    if evaluator is None:
        evaluator = GraphSSLEvaluator()
    
    report = {}
    
    # Node classification
    try:
        report["node_classification"] = evaluator.evaluate_node_classification(model, data)
    except Exception as e:
        print(f"Node classification evaluation failed: {e}")
        report["node_classification"] = {}
    
    # Link prediction
    try:
        report["link_prediction"] = evaluator.evaluate_link_prediction(model, data)
    except Exception as e:
        print(f"Link prediction evaluation failed: {e}")
        report["link_prediction"] = {}
    
    # Clustering
    try:
        report["clustering"] = evaluator.evaluate_clustering(model, data)
    except Exception as e:
        print(f"Clustering evaluation failed: {e}")
        report["clustering"] = {}
    
    # Reconstruction quality
    try:
        report["reconstruction"] = evaluator.evaluate_reconstruction_quality(model, data)
    except Exception as e:
        print(f"Reconstruction evaluation failed: {e}")
        report["reconstruction"] = {}
    
    return report
