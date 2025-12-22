"""Unit tests for graph self-supervised learning models."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.encoders import GCNEncoder, AttributeDecoder, ContrastiveDecoder
from src.models.ssl_models import NodeMaskingSSL, ContrastiveSSL, GraphCLSSL
from src.utils.device import get_device, set_seed
from src.utils.data_utils import mask_node_features, create_positive_pairs, create_negative_pairs


@pytest.fixture
def sample_data():
    """Create sample graph data for testing."""
    set_seed(42)
    
    # Create a small graph
    num_nodes = 50
    num_features = 10
    num_edges = 100
    
    # Random features
    x = torch.randn(num_nodes, num_features)
    
    # Random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Random labels
    y = torch.randint(0, 3, (num_nodes,))
    
    return Data(x=x, edge_index=edge_index, y=y)


@pytest.fixture
def sample_encoder():
    """Create sample encoder for testing."""
    return GCNEncoder(
        in_channels=10,
        hidden_channels=32,
        out_channels=16,
        num_layers=2,
        dropout=0.5
    )


@pytest.fixture
def sample_decoder():
    """Create sample decoder for testing."""
    return AttributeDecoder(
        in_channels=16,
        out_channels=10,
        hidden_channels=32
    )


class TestGCNEncoder:
    """Test GCN encoder."""
    
    def test_forward_pass(self, sample_encoder, sample_data):
        """Test forward pass of encoder."""
        encoder = sample_encoder
        data = sample_data
        
        embeddings = encoder(data.x, data.edge_index)
        
        assert embeddings.shape == (data.num_nodes, 16)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
    
    def test_different_configurations(self):
        """Test encoder with different configurations."""
        # Test with batch normalization
        encoder_bn = GCNEncoder(10, 32, 16, use_batch_norm=True)
        encoder_no_bn = GCNEncoder(10, 32, 16, use_batch_norm=False)
        
        data = Data(x=torch.randn(20, 10), edge_index=torch.randint(0, 20, (2, 30)))
        
        emb_bn = encoder_bn(data.x, data.edge_index)
        emb_no_bn = encoder_no_bn(data.x, data.edge_index)
        
        assert emb_bn.shape == emb_no_bn.shape
        assert not torch.equal(emb_bn, emb_no_bn)  # Should be different
    
    def test_residual_connections(self):
        """Test residual connections."""
        encoder_res = GCNEncoder(10, 32, 10, use_residual=True)
        encoder_no_res = GCNEncoder(10, 32, 10, use_residual=False)
        
        data = Data(x=torch.randn(20, 10), edge_index=torch.randint(0, 20, (2, 30)))
        
        emb_res = encoder_res(data.x, data.edge_index)
        emb_no_res = encoder_no_res(data.x, data.edge_index)
        
        assert emb_res.shape == emb_no_res.shape


class TestAttributeDecoder:
    """Test attribute decoder."""
    
    def test_forward_pass(self, sample_decoder):
        """Test forward pass of decoder."""
        decoder = sample_decoder
        embeddings = torch.randn(20, 16)
        
        reconstructed = decoder(embeddings)
        
        assert reconstructed.shape == (20, 10)
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()
    
    def test_different_layers(self):
        """Test decoder with different number of layers."""
        decoder_1 = AttributeDecoder(16, 10, num_layers=1)
        decoder_2 = AttributeDecoder(16, 10, num_layers=2)
        decoder_3 = AttributeDecoder(16, 10, num_layers=3)
        
        embeddings = torch.randn(20, 16)
        
        out_1 = decoder_1(embeddings)
        out_2 = decoder_2(embeddings)
        out_3 = decoder_3(embeddings)
        
        assert out_1.shape == out_2.shape == out_3.shape == (20, 10)


class TestContrastiveDecoder:
    """Test contrastive decoder."""
    
    def test_forward_pass(self):
        """Test forward pass of contrastive decoder."""
        decoder = ContrastiveDecoder(16, hidden_channels=32)
        embeddings = torch.randn(20, 16)
        
        scores = decoder(embeddings)
        
        assert scores.shape == (20, 1)
        assert not torch.isnan(scores).any()
    
    def test_similarity_computation(self):
        """Test similarity computation."""
        decoder = ContrastiveDecoder(16, hidden_channels=32)
        
        x1 = torch.randn(10, 16)
        x2 = torch.randn(10, 16)
        
        similarities = decoder.compute_similarity(x1, x2)
        
        assert similarities.shape == (10,)
        assert not torch.isnan(similarities).any()


class TestNodeMaskingSSL:
    """Test node masking SSL model."""
    
    def test_forward_pass(self, sample_encoder, sample_decoder, sample_data):
        """Test forward pass of node masking model."""
        model = NodeMaskingSSL(sample_encoder, sample_decoder)
        data = sample_data
        
        embeddings, reconstructed, targets = model(data)
        
        assert embeddings.shape == (data.num_nodes, 16)
        assert reconstructed.shape[1] == data.num_node_features
        assert targets.shape[1] == data.num_node_features
        assert reconstructed.shape[0] == targets.shape[0]
    
    def test_loss_computation(self, sample_encoder, sample_decoder, sample_data):
        """Test loss computation."""
        model = NodeMaskingSSL(sample_encoder, sample_decoder)
        data = sample_data
        
        embeddings, reconstructed, targets = model(data)
        loss = model.compute_loss(reconstructed, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_different_mask_ratios(self, sample_encoder, sample_decoder, sample_data):
        """Test with different mask ratios."""
        data = sample_data
        
        model_1 = NodeMaskingSSL(sample_encoder, sample_decoder, mask_ratio=0.1)
        model_2 = NodeMaskingSSL(sample_encoder, sample_decoder, mask_ratio=0.5)
        
        _, recon_1, targets_1 = model_1(data)
        _, recon_2, targets_2 = model_2(data)
        
        # Different mask ratios should result in different numbers of masked nodes
        assert recon_1.shape[0] != recon_2.shape[0]
        assert targets_1.shape[0] != targets_2.shape[0]


class TestContrastiveSSL:
    """Test contrastive SSL model."""
    
    def test_forward_pass(self, sample_encoder, sample_data):
        """Test forward pass of contrastive model."""
        contrastive_decoder = ContrastiveDecoder(16, hidden_channels=32)
        model = ContrastiveSSL(sample_encoder, contrastive_decoder)
        data = sample_data
        
        embeddings, pos_scores, neg_scores = model(data)
        
        assert embeddings.shape == (data.num_nodes, 16)
        assert pos_scores.shape[0] > 0
        assert neg_scores.shape[0] > 0
        assert pos_scores.shape == neg_scores.shape
    
    def test_loss_computation(self, sample_encoder, sample_data):
        """Test loss computation."""
        contrastive_decoder = ContrastiveDecoder(16, hidden_channels=32)
        model = ContrastiveSSL(sample_encoder, contrastive_decoder)
        data = sample_data
        
        embeddings, pos_scores, neg_scores = model(data)
        loss = model.compute_loss(pos_scores, neg_scores)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestGraphCLSSL:
    """Test GraphCL SSL model."""
    
    def test_forward_pass(self, sample_encoder, sample_data):
        """Test forward pass of GraphCL model."""
        contrastive_decoder = ContrastiveDecoder(16, hidden_channels=32)
        model = GraphCLSSL(sample_encoder, contrastive_decoder)
        data = sample_data
        
        orig_embeddings, aug_embeddings = model(data)
        
        assert orig_embeddings.shape == (data.num_nodes, 16)
        assert aug_embeddings.shape == (data.num_nodes, 16)
        assert orig_embeddings.shape == aug_embeddings.shape
    
    def test_loss_computation(self, sample_encoder, sample_data):
        """Test loss computation."""
        contrastive_decoder = ContrastiveDecoder(16, hidden_channels=32)
        model = GraphCLSSL(sample_encoder, contrastive_decoder)
        data = sample_data
        
        orig_embeddings, aug_embeddings = model(data)
        loss = model.compute_loss(orig_embeddings, aug_embeddings)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestDataUtils:
    """Test data utility functions."""
    
    def test_mask_node_features(self, sample_data):
        """Test node feature masking."""
        data = sample_data
        mask_ratio = 0.3
        
        masked_x, mask_indices = mask_node_features(data.x, mask_ratio=mask_ratio)
        
        assert masked_x.shape == data.x.shape
        assert mask_indices.shape[0] <= int(mask_ratio * data.num_nodes)
        assert torch.all(masked_x[mask_indices] == 0)
    
    def test_create_positive_pairs(self, sample_data):
        """Test positive pair creation."""
        data = sample_data
        num_pairs = 50
        
        pos_pairs = create_positive_pairs(data.edge_index, num_pairs=num_pairs)
        
        assert pos_pairs.shape == (2, min(num_pairs, data.num_edges))
        assert torch.all(pos_pairs >= 0)
        assert torch.all(pos_pairs < data.num_nodes)
    
    def test_create_negative_pairs(self, sample_data):
        """Test negative pair creation."""
        data = sample_data
        num_pairs = 50
        
        neg_pairs = create_negative_pairs(data.num_nodes, num_pairs=num_pairs)
        
        assert neg_pairs.shape == (2, num_pairs)
        assert torch.all(neg_pairs >= 0)
        assert torch.all(neg_pairs < data.num_nodes)
        # Ensure no self-loops
        assert torch.all(neg_pairs[0] != neg_pairs[1])


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that random numbers are reproducible
        torch.manual_seed(42)
        rand1 = torch.randn(10)
        
        set_seed(42)
        torch.manual_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.allclose(rand1, rand2)


if __name__ == "__main__":
    pytest.main([__file__])
