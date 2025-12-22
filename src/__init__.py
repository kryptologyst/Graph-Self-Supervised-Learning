"""Graph Self-Supervised Learning package."""

__version__ = "1.0.0"
__author__ = "AI Projects"
__email__ = "ai@example.com"

from .models.encoders import GCNEncoder, AttributeDecoder, ContrastiveDecoder
from .models.ssl_models import NodeMaskingSSL, ContrastiveSSL, GraphCLSSL, MultiTaskSSL
from .utils.device import get_device, set_seed
from .utils.data_utils import mask_node_features, create_positive_pairs, create_negative_pairs

__all__ = [
    "GCNEncoder",
    "AttributeDecoder", 
    "ContrastiveDecoder",
    "NodeMaskingSSL",
    "ContrastiveSSL",
    "GraphCLSSL",
    "MultiTaskSSL",
    "get_device",
    "set_seed",
    "mask_node_features",
    "create_positive_pairs",
    "create_negative_pairs",
]
