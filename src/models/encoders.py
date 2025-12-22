"""Enhanced GCN encoder for self-supervised learning."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder with normalization and residual connections.
    
    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden features.
        out_channels: Number of output features.
        num_layers: Number of GCN layers.
        dropout: Dropout rate.
        use_batch_norm: Whether to use batch normalization.
        use_residual: Whether to use residual connections.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        use_residual: bool = False
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.dropout = dropout
        
        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(out_channels))
        
        # Projection for residual connections
        if use_residual and in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node feature matrix.
            edge_index: Edge index tensor.
            edge_weight: Optional edge weights.
            
        Returns:
            Node embeddings.
        """
        residual = x
        
        for i, conv in enumerate(self.convs):
            # Apply convolution
            x = conv(x, edge_index, edge_weight)
            
            # Apply batch normalization
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Apply activation (except for last layer)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual
        
        return x


class AttributeDecoder(nn.Module):
    """Decoder for reconstructing node attributes.
    
    Args:
        in_channels: Number of input features (embedding dimension).
        out_channels: Number of output features (original feature dimension).
        hidden_channels: Number of hidden features.
        num_layers: Number of layers.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = max(in_channels, out_channels)
        
        layers = []
        current_channels = in_channels
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_channels = hidden_channels
        
        # Output layer
        layers.append(nn.Linear(current_channels, out_channels))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node embeddings.
            
        Returns:
            Reconstructed node features.
        """
        return self.decoder(x)


class ContrastiveDecoder(nn.Module):
    """Decoder for contrastive learning tasks.
    
    Args:
        in_channels: Number of input features (embedding dimension).
        hidden_channels: Number of hidden features.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node embeddings.
            
        Returns:
            Contrastive scores.
        """
        return self.decoder(x)
    
    def compute_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between two sets of embeddings.
        
        Args:
            x1: First set of embeddings.
            x2: Second set of embeddings.
            
        Returns:
            Similarity scores.
        """
        # Concatenate embeddings
        combined = torch.cat([x1, x2], dim=-1)
        return self.decoder(combined).squeeze(-1)
