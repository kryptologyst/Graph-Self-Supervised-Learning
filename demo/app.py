"""Interactive demo for Graph Self-Supervised Learning."""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import json

from src.models.encoders import GCNEncoder, AttributeDecoder, ContrastiveDecoder
from src.models.ssl_models import NodeMaskingSSL, ContrastiveSSL, GraphCLSSL
from src.data.datasets import load_dataset, create_train_val_test_splits, preprocess_graph
from src.utils.device import get_device, set_seed
from src.eval.evaluator import GraphSSLEvaluator


def load_model_and_data(model_path: str, data_path: str):
    """Load trained model and data."""
    try:
        # Load data
        data = torch.load(data_path)
        
        # Load model (simplified - in practice you'd need to reconstruct the model)
        # For demo purposes, we'll create a simple model
        model = NodeMaskingSSL(
            encoder=GCNEncoder(data['x'].size(1), 64, 64),
            decoder=AttributeDecoder(64, data['x'].size(1))
        )
        
        return model, data
    except Exception as e:
        st.error(f"Error loading model/data: {e}")
        return None, None


def visualize_embeddings(embeddings, labels=None, title="Node Embeddings"):
    """Create t-SNE visualization of embeddings."""
    from sklearn.manifold import TSNE
    
    # Reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels if labels is not None else 'Unknown'
    })
    
    # Create plot
    fig = px.scatter(
        df, x='x', y='y', color='label',
        title=title,
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'}
    )
    
    return fig


def visualize_graph_structure(data, max_nodes=100):
    """Visualize graph structure."""
    if data['x'].size(0) > max_nodes:
        st.warning(f"Graph has {data['x'].size(0)} nodes. Showing first {max_nodes} nodes.")
        node_mask = torch.zeros(data['x'].size(0), dtype=torch.bool)
        node_mask[:max_nodes] = True
        subgraph_x = data['x'][node_mask]
        subgraph_edge_index = data['edge_index'][:, 
            torch.isin(data['edge_index'][0], torch.where(node_mask)[0]) &
            torch.isin(data['edge_index'][1], torch.where(node_mask)[0])
        ]
    else:
        subgraph_x = data['x']
        subgraph_edge_index = data['edge_index']
    
    # Create network visualization
    fig = go.Figure()
    
    # Add edges
    for i in range(subgraph_edge_index.size(1)):
        src, dst = subgraph_edge_index[:, i]
        fig.add_trace(go.Scatter(
            x=[subgraph_x[src, 0].item(), subgraph_x[dst, 0].item()],
            y=[subgraph_x[src, 1].item(), subgraph_x[dst, 1].item()],
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=subgraph_x[:, 0].numpy(),
        y=subgraph_x[:, 1].numpy(),
        mode='markers',
        marker=dict(
            size=8,
            color=subgraph_x[:, 2].numpy() if subgraph_x.size(1) > 2 else 'blue',
            colorscale='Viridis',
            showscale=True
        ),
        text=[f"Node {i}" for i in range(subgraph_x.size(0))],
        hovertemplate="%{text}<br>Feature 1: %{x:.3f}<br>Feature 2: %{y:.3f}<extra></extra>",
        name="Nodes"
    ))
    
    fig.update_layout(
        title="Graph Structure",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        showlegend=False
    )
    
    return fig


def main():
    """Main demo application."""
    st.set_page_config(
        page_title="Graph Self-Supervised Learning Demo",
        page_icon="🕸️",
        layout="wide"
    )
    
    st.title("🕸️ Graph Self-Supervised Learning Demo")
    st.markdown("Explore graph neural networks with self-supervised learning techniques")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_options = {
        "Cora": "cora",
        "Citeseer": "citeseer", 
        "Pubmed": "pubmed",
        "Synthetic SBM": "synthetic_sbm_1000"
    }
    
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=list(dataset_options.keys()),
        index=0
    )
    
    # Model type
    model_options = {
        "Node Masking": "node_masking",
        "Contrastive Learning": "contrastive",
        "GraphCL": "graphcl"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model Type",
        options=list(model_options.keys()),
        index=0
    )
    
    # Parameters
    st.sidebar.subheader("Parameters")
    mask_ratio = st.sidebar.slider("Mask Ratio", 0.1, 0.5, 0.3, 0.05)
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
    num_epochs = st.sidebar.slider("Epochs", 10, 200, 50, 10)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Graph Visualization")
        
        # Load and preprocess data
        try:
            data, dataset_info = load_dataset(dataset_options[selected_dataset])
            data = preprocess_graph(data)
            data = create_train_val_test_splits(data, random_seed=42)
            
            st.success(f"Loaded {dataset_info}")
            st.info(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_node_features}")
            
            # Visualize graph structure
            if data.num_nodes <= 1000:  # Only for smaller graphs
                fig_graph = visualize_graph_structure({
                    'x': data.x,
                    'edge_index': data.edge_index
                })
                st.plotly_chart(fig_graph, use_container_width=True)
            else:
                st.info("Graph too large for structure visualization. Showing embeddings instead.")
                
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return
    
    with col2:
        st.header("Model Information")
        
        # Create model
        try:
            encoder = GCNEncoder(data.num_node_features, 64, 64)
            decoder = AttributeDecoder(64, data.num_node_features)
            
            if model_options[selected_model] == "node_masking":
                model = NodeMaskingSSL(encoder, decoder, mask_ratio=mask_ratio)
            elif model_options[selected_model] == "contrastive":
                contrastive_decoder = ContrastiveDecoder(64)
                model = ContrastiveSSL(encoder, contrastive_decoder)
            elif model_options[selected_model] == "graphcl":
                contrastive_decoder = ContrastiveDecoder(64)
                model = GraphCLSSL(encoder, contrastive_decoder)
            
            st.success(f"Created {selected_model} model")
            st.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            st.error(f"Error creating model: {e}")
            return
    
    # Training section
    st.header("Training")
    
    if st.button("🚀 Train Model", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Set up training
        device = get_device()
        model.to(device)
        data = data.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        losses = []
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, NodeMaskingSSL):
                embeddings, reconstructed, targets = model(data)
                loss = model.compute_loss(reconstructed, targets)
            elif isinstance(model, ContrastiveSSL):
                embeddings, pos_scores, neg_scores = model(data)
                loss = model.compute_loss(pos_scores, neg_scores)
            elif isinstance(model, GraphCLSSL):
                orig_embeddings, aug_embeddings = model(data)
                loss = model.compute_loss(orig_embeddings, aug_embeddings)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Update progress
            progress = (epoch + 1) / num_epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}")
        
        st.success("Training completed!")
        
        # Plot loss curve
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=list(range(len(losses))),
            y=losses,
            mode='lines',
            name='Training Loss'
        ))
        fig_loss.update_layout(
            title="Training Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss"
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Evaluation section
    st.header("Evaluation")
    
    if st.button("📊 Evaluate Model"):
        try:
            # Get embeddings
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'encoder'):
                    embeddings = model.encoder(data.x, data.edge_index)
                else:
                    embeddings = model(data.x, data.edge_index)
            
            # Create evaluator
            evaluator = GraphSSLEvaluator(device=device)
            
            # Evaluate different tasks
            col_eval1, col_eval2 = st.columns(2)
            
            with col_eval1:
                st.subheader("Node Classification")
                try:
                    nc_metrics = evaluator.evaluate_node_classification(model, data)
                    for metric, value in nc_metrics.items():
                        st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
                except Exception as e:
                    st.error(f"Node classification failed: {e}")
            
            with col_eval2:
                st.subheader("Link Prediction")
                try:
                    lp_metrics = evaluator.evaluate_link_prediction(model, data)
                    for metric, value in lp_metrics.items():
                        st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
                except Exception as e:
                    st.error(f"Link prediction failed: {e}")
            
            # Embedding visualization
            st.subheader("Embedding Visualization")
            if hasattr(data, 'y') and data.y is not None:
                labels = data.y.cpu().numpy()
            else:
                labels = None
            
            fig_emb = visualize_embeddings(
                embeddings.cpu().numpy(), 
                labels,
                f"{selected_model} Embeddings"
            )
            st.plotly_chart(fig_emb, use_container_width=True)
            
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
    
    # Node exploration
    st.header("Node Exploration")
    
    if st.button("🔍 Explore Nodes"):
        try:
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'encoder'):
                    embeddings = model.encoder(data.x, data.edge_index)
                else:
                    embeddings = model(data.x, data.edge_index)
            
            # Select a random node
            node_id = st.selectbox(
                "Select Node",
                options=list(range(min(100, data.num_nodes))),
                index=0
            )
            
            col_node1, col_node2 = st.columns(2)
            
            with col_node1:
                st.subheader(f"Node {node_id} Information")
                st.write(f"**Features:** {data.x[node_id].cpu().numpy()}")
                if hasattr(data, 'y') and data.y is not None:
                    st.write(f"**Label:** {data.y[node_id].item()}")
                st.write(f"**Embedding:** {embeddings[node_id].cpu().numpy()}")
            
            with col_node2:
                st.subheader("Neighbors")
                # Find neighbors
                neighbors = data.edge_index[1][data.edge_index[0] == node_id]
                if len(neighbors) > 0:
                    st.write(f"**Number of neighbors:** {len(neighbors)}")
                    st.write(f"**Neighbor IDs:** {neighbors.cpu().numpy()[:10]}")  # Show first 10
                else:
                    st.write("No neighbors found")
            
        except Exception as e:
            st.error(f"Node exploration failed: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and PyTorch Geometric")


if __name__ == "__main__":
    main()
