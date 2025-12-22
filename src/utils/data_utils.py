"""Data utilities for graph self-supervised learning."""

import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def mask_node_features(
    x: torch.Tensor, 
    mask_ratio: float = 0.3,
    mask_value: float = 0.0,
    random_seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask node features for self-supervised learning.
    
    Args:
        x: Node feature matrix of shape [num_nodes, num_features].
        mask_ratio: Fraction of nodes to mask.
        mask_value: Value to use for masking.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (masked_features, mask_indices).
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    num_nodes = x.size(0)
    num_mask = int(mask_ratio * num_nodes)
    
    # Randomly select nodes to mask
    mask_indices = torch.randperm(num_nodes)[:num_mask]
    
    # Create masked version
    masked_x = x.clone()
    masked_x[mask_indices] = mask_value
    
    return masked_x, mask_indices


def add_noise_to_features(
    x: torch.Tensor,
    noise_ratio: float = 0.1,
    noise_std: float = 0.1,
    random_seed: Optional[int] = None
) -> torch.Tensor:
    """Add Gaussian noise to node features.
    
    Args:
        x: Node feature matrix.
        noise_ratio: Fraction of features to add noise to.
        noise_std: Standard deviation of noise.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Noisy feature matrix.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    noisy_x = x.clone()
    num_features = x.size(1)
    num_noisy_features = int(noise_ratio * num_features)
    
    # Select random features to add noise to
    feature_indices = torch.randperm(num_features)[:num_noisy_features]
    
    # Add Gaussian noise
    noise = torch.randn_like(noisy_x[:, feature_indices]) * noise_std
    noisy_x[:, feature_indices] += noise
    
    return noisy_x


def edge_dropout(
    edge_index: torch.Tensor,
    dropout_ratio: float = 0.1,
    random_seed: Optional[int] = None
) -> torch.Tensor:
    """Randomly drop edges from the graph.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges].
        dropout_ratio: Fraction of edges to drop.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Edge index with dropped edges.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    num_edges = edge_index.size(1)
    num_keep = int((1 - dropout_ratio) * num_edges)
    
    # Randomly select edges to keep
    keep_indices = torch.randperm(num_edges)[:num_keep]
    
    return edge_index[:, keep_indices]


def create_positive_pairs(
    edge_index: torch.Tensor,
    num_pairs: int = 1000,
    random_seed: Optional[int] = None
) -> torch.Tensor:
    """Create positive node pairs for contrastive learning.
    
    Args:
        edge_index: Edge index tensor.
        num_pairs: Number of positive pairs to create.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Tensor of shape [2, num_pairs] containing positive pairs.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    num_edges = edge_index.size(1)
    num_pairs = min(num_pairs, num_edges)
    
    # Randomly select edges as positive pairs
    pair_indices = torch.randperm(num_edges)[:num_pairs]
    
    return edge_index[:, pair_indices]


def create_negative_pairs(
    num_nodes: int,
    num_pairs: int = 1000,
    random_seed: Optional[int] = None
) -> torch.Tensor:
    """Create negative node pairs for contrastive learning.
    
    Args:
        num_nodes: Total number of nodes in the graph.
        num_pairs: Number of negative pairs to create.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Tensor of shape [2, num_pairs] containing negative pairs.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Randomly sample node pairs
    src_nodes = torch.randint(0, num_nodes, (num_pairs,))
    dst_nodes = torch.randint(0, num_nodes, (num_pairs,))
    
    # Ensure src != dst
    same_node_mask = src_nodes == dst_nodes
    dst_nodes[same_node_mask] = (dst_nodes[same_node_mask] + 1) % num_nodes
    
    return torch.stack([src_nodes, dst_nodes])


def get_graph_statistics(data: Data) -> Dict[str, Union[int, float]]:
    """Get basic statistics about the graph.
    
    Args:
        data: PyTorch Geometric Data object.
        
    Returns:
        Dictionary containing graph statistics.
    """
    stats = {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_node_features,
        "avg_degree": (2 * data.num_edges) / data.num_nodes,
        "density": (2 * data.num_edges) / (data.num_nodes * (data.num_nodes - 1)),
    }
    
    if hasattr(data, "y") and data.y is not None:
        stats["num_classes"] = len(torch.unique(data.y))
    
    return stats


def visualize_graph(
    data: Data,
    max_nodes: int = 100,
    save_path: Optional[str] = None
) -> None:
    """Visualize a small graph using NetworkX.
    
    Args:
        data: PyTorch Geometric Data object.
        max_nodes: Maximum number of nodes to visualize.
        save_path: Optional path to save the visualization.
    """
    if data.num_nodes > max_nodes:
        print(f"Graph has {data.num_nodes} nodes, only visualizing first {max_nodes}")
        # Subgraph sampling
        node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        node_mask[:max_nodes] = True
        subgraph = data.subgraph(node_mask)
    else:
        subgraph = data
    
    # Convert to NetworkX
    G = to_networkx(subgraph, to_undirected=True)
    
    # Basic visualization
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    pos = torch.randn(subgraph.num_nodes, 2).numpy()
    
    # Draw nodes
    plt.scatter(pos[:, 0], pos[:, 1], c='lightblue', s=100, alpha=0.7)
    
    # Draw edges
    for edge in subgraph.edge_index.t().numpy():
        plt.plot([pos[edge[0], 0], pos[edge[1], 0]], 
                [pos[edge[0], 1], pos[edge[1], 1]], 
                'k-', alpha=0.3, linewidth=0.5)
    
    plt.title(f"Graph Visualization ({subgraph.num_nodes} nodes, {subgraph.num_edges} edges)")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
