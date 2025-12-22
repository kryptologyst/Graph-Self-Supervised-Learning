"""Self-supervised learning models for graphs."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .encoders import GCNEncoder, AttributeDecoder, ContrastiveDecoder
from ..utils.data_utils import (
    mask_node_features, 
    add_noise_to_features, 
    edge_dropout,
    create_positive_pairs,
    create_negative_pairs
)


class NodeMaskingSSL(nn.Module):
    """Node attribute masking for self-supervised learning.
    
    This implements a BERT-style masking approach where node features
    are randomly masked and the model learns to reconstruct them.
    """
    
    def __init__(
        self,
        encoder: GCNEncoder,
        decoder: AttributeDecoder,
        mask_ratio: float = 0.3,
        mask_value: float = 0.0
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        
    def forward(
        self, 
        data: Data,
        mask_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for node masking.
        
        Args:
            data: Graph data object.
            mask_indices: Optional pre-computed mask indices.
            
        Returns:
            Tuple of (embeddings, reconstructed_features, target_features).
        """
        # Create masked features
        if mask_indices is None:
            masked_x, mask_indices = mask_node_features(
                data.x, 
                mask_ratio=self.mask_ratio,
                mask_value=self.mask_value
            )
        else:
            masked_x = data.x.clone()
            masked_x[mask_indices] = self.mask_value
        
        # Encode
        embeddings = self.encoder(masked_x, data.edge_index)
        
        # Decode only masked nodes
        reconstructed = self.decoder(embeddings[mask_indices])
        targets = data.x[mask_indices]
        
        return embeddings, reconstructed, targets
    
    def compute_loss(
        self, 
        reconstructed: torch.Tensor, 
        targets: torch.Tensor,
        loss_fn: nn.Module = nn.MSELoss()
    ) -> torch.Tensor:
        """Compute reconstruction loss.
        
        Args:
            reconstructed: Reconstructed features.
            targets: Target features.
            loss_fn: Loss function.
            
        Returns:
            Reconstruction loss.
        """
        return loss_fn(reconstructed, targets)


class ContrastiveSSL(nn.Module):
    """Contrastive self-supervised learning for graphs.
    
    This implements a contrastive approach where the model learns to
    distinguish between positive and negative node pairs.
    """
    
    def __init__(
        self,
        encoder: GCNEncoder,
        decoder: ContrastiveDecoder,
        temperature: float = 0.1
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = temperature
        
    def forward(
        self, 
        data: Data,
        positive_pairs: Optional[torch.Tensor] = None,
        negative_pairs: Optional[torch.Tensor] = None,
        num_pairs: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for contrastive learning.
        
        Args:
            data: Graph data object.
            positive_pairs: Optional pre-computed positive pairs.
            negative_pairs: Optional pre-computed negative pairs.
            num_pairs: Number of pairs to sample.
            
        Returns:
            Tuple of (embeddings, positive_scores, negative_scores).
        """
        # Encode
        embeddings = self.encoder(data.x, data.edge_index)
        
        # Create pairs if not provided
        if positive_pairs is None:
            positive_pairs = create_positive_pairs(data.edge_index, num_pairs)
        if negative_pairs is None:
            negative_pairs = create_negative_pairs(data.num_nodes, num_pairs)
        
        # Compute similarities
        pos_scores = self.decoder.compute_similarity(
            embeddings[positive_pairs[0]], 
            embeddings[positive_pairs[1]]
        )
        neg_scores = self.decoder.compute_similarity(
            embeddings[negative_pairs[0]], 
            embeddings[negative_pairs[1]]
        )
        
        return embeddings, pos_scores, neg_scores
    
    def compute_loss(
        self, 
        pos_scores: torch.Tensor, 
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            pos_scores: Positive pair scores.
            neg_scores: Negative pair scores.
            
        Returns:
            Contrastive loss.
        """
        # InfoNCE loss
        pos_exp = torch.exp(pos_scores / self.temperature)
        neg_exp = torch.exp(neg_scores / self.temperature)
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp.sum()))
        return loss.mean()


class GraphCLSSL(nn.Module):
    """Graph Contrastive Learning (GraphCL) for self-supervised learning.
    
    This implements the GraphCL approach with multiple augmentation strategies.
    """
    
    def __init__(
        self,
        encoder: GCNEncoder,
        decoder: ContrastiveDecoder,
        temperature: float = 0.1,
        aug_ratio: float = 0.1
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.temperature = temperature
        self.aug_ratio = aug_ratio
        
    def augment_graph(self, data: Data) -> Data:
        """Apply graph augmentation.
        
        Args:
            data: Original graph data.
            
        Returns:
            Augmented graph data.
        """
        # Create augmented version
        aug_data = data.clone()
        
        # Feature masking
        aug_data.x, _ = mask_node_features(
            data.x, 
            mask_ratio=self.aug_ratio,
            mask_value=0.0
        )
        
        # Edge dropout
        aug_data.edge_index = edge_dropout(
            data.edge_index, 
            dropout_ratio=self.aug_ratio
        )
        
        return aug_data
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for GraphCL.
        
        Args:
            data: Graph data object.
            
        Returns:
            Tuple of (original_embeddings, augmented_embeddings).
        """
        # Original graph
        orig_embeddings = self.encoder(data.x, data.edge_index)
        
        # Augmented graph
        aug_data = self.augment_graph(data)
        aug_embeddings = self.encoder(aug_data.x, aug_data.edge_index)
        
        return orig_embeddings, aug_embeddings
    
    def compute_loss(
        self, 
        orig_embeddings: torch.Tensor, 
        aug_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute GraphCL loss.
        
        Args:
            orig_embeddings: Original graph embeddings.
            aug_embeddings: Augmented graph embeddings.
            
        Returns:
            GraphCL loss.
        """
        # Normalize embeddings
        orig_embeddings = F.normalize(orig_embeddings, p=2, dim=1)
        aug_embeddings = F.normalize(aug_embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(orig_embeddings, aug_embeddings.t()) / self.temperature
        
        # Labels (diagonal elements are positive pairs)
        labels = torch.arange(orig_embeddings.size(0)).to(orig_embeddings.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class MultiTaskSSL(nn.Module):
    """Multi-task self-supervised learning combining multiple SSL objectives.
    
    This combines node masking, contrastive learning, and other SSL tasks.
    """
    
    def __init__(
        self,
        encoder: GCNEncoder,
        attribute_decoder: AttributeDecoder,
        contrastive_decoder: ContrastiveDecoder,
        mask_ratio: float = 0.3,
        temperature: float = 0.1,
        task_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.attribute_decoder = attribute_decoder
        self.contrastive_decoder = contrastive_decoder
        
        # SSL modules
        self.node_masking = NodeMaskingSSL(
            encoder, attribute_decoder, mask_ratio
        )
        self.contrastive = ContrastiveSSL(
            encoder, contrastive_decoder, temperature
        )
        self.graphcl = GraphCLSSL(
            encoder, contrastive_decoder, temperature
        )
        
        # Task weights
        self.task_weights = task_weights or {
            "masking": 1.0,
            "contrastive": 0.5,
            "graphcl": 0.5
        }
    
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task SSL.
        
        Args:
            data: Graph data object.
            
        Returns:
            Dictionary containing outputs from different tasks.
        """
        outputs = {}
        
        # Node masking
        embeddings, reconstructed, targets = self.node_masking(data)
        outputs["masking"] = {
            "embeddings": embeddings,
            "reconstructed": reconstructed,
            "targets": targets
        }
        
        # Contrastive learning
        embeddings, pos_scores, neg_scores = self.contrastive(data)
        outputs["contrastive"] = {
            "embeddings": embeddings,
            "pos_scores": pos_scores,
            "neg_scores": neg_scores
        }
        
        # GraphCL
        orig_embeddings, aug_embeddings = self.graphcl(data)
        outputs["graphcl"] = {
            "orig_embeddings": orig_embeddings,
            "aug_embeddings": aug_embeddings
        }
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute combined multi-task loss.
        
        Args:
            outputs: Outputs from forward pass.
            
        Returns:
            Combined loss.
        """
        total_loss = 0.0
        
        # Node masking loss
        masking_loss = self.node_masking.compute_loss(
            outputs["masking"]["reconstructed"],
            outputs["masking"]["targets"]
        )
        total_loss += self.task_weights["masking"] * masking_loss
        
        # Contrastive loss
        contrastive_loss = self.contrastive.compute_loss(
            outputs["contrastive"]["pos_scores"],
            outputs["contrastive"]["neg_scores"]
        )
        total_loss += self.task_weights["contrastive"] * contrastive_loss
        
        # GraphCL loss
        graphcl_loss = self.graphcl.compute_loss(
            outputs["graphcl"]["orig_embeddings"],
            outputs["graphcl"]["aug_embeddings"]
        )
        total_loss += self.task_weights["graphcl"] * graphcl_loss
        
        return total_loss
