import torch.nn as nn
from lib_mpe.models.encoder.embeddings.learnable_per_node import (
    LearnablePerNodeEmbedding,
)
from lib_mpe.models.encoder.embeddings.learnable_per_node_value import (
    LearnablePerNodeValueEmbedding,
)
from loguru import logger
from torch_geometric.nn import HypergraphConv, global_mean_pool


class HGNNLayer(nn.Module):
    def __init__(self, hgnn_layer, bn_layer, dropout_layer, skip_connection=None):
        super().__init__()
        self.hgnn_layer = hgnn_layer
        self.bn_layer = bn_layer
        self.dropout_layer = dropout_layer
        self.skip_connection = skip_connection

    def forward(self, x, hyperedge_index, hyperedge_attr):
        if self.skip_connection:
            residual = self.skip_connection(x)

        new_features = self.hgnn_layer(
            x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr
        )
        new_features = self.bn_layer(new_features)
        new_features = self.dropout_layer(new_features)

        if self.skip_connection:
            return new_features + residual
        return new_features


class HGNNEmbeddingLayer(nn.Module):
    HYPERGRAPH_CLASSES = {
        "HypergraphConv": HypergraphConv,
    }

    def __init__(
        self,
        num_nodes,
        encoder_channels,
        initialize_embed_size,
        pgm_max_clique_size,
        cfg,
        device,
    ):
        super().__init__()
        self.debug = cfg.debug
        self.embedding_size = int(encoder_channels[-1])
        self.use_residual = cfg.encoder_residual_connections
        self.hgnn_initial_embedding_type = cfg.hgnn_initial_embedding_type
        self.hgnn_attention_mode = cfg.hgnn_attention_mode
        if self.use_residual:
            logger.info("Using residual connections in the encoder model.")

        self.ip_encoding_size = 2**pgm_max_clique_size
        self.node_embedding_layer = nn.Linear(
            initialize_embed_size, self.ip_encoding_size
        )
        encoder_channels = [self.ip_encoding_size] + encoder_channels
        # create a learnable embedding for the initial layer for each node with size: ip_encoding_size
        if self.hgnn_initial_embedding_type == "learnable_per_node":
            logger.info("Using learnable per node embedding.")
            self.node_embedding_layer_learnable = LearnablePerNodeEmbedding(
                num_nodes, self.ip_encoding_size, device
            )
            self.node_embedding_layer_learnable.to(device)
        elif self.hgnn_initial_embedding_type == "learnable_per_node_value":
            logger.info("Using learnable per node value embedding.")
            self.node_embedding_layer_learnable = LearnablePerNodeValueEmbedding(
                num_nodes, self.ip_encoding_size, device
            )
            self.node_embedding_layer_learnable.to(device)
        else:
            logger.info("Not using learnable embedding.")

        layers = []
        for in_channels, out_channels in zip(
            encoder_channels[:-1], encoder_channels[1:]
        ):
            hgnn_layer = self._get_hgnn_layer(cfg, in_channels, out_channels)
            bn_layer = (
                nn.BatchNorm1d(out_channels) if not cfg.no_batchnorm else nn.Identity()
            )
            dropout_layer = (
                nn.Dropout(cfg.dropout_rate) if not cfg.no_dropout else nn.Identity()
            )

            skip_connection = None
            if self.use_residual:
                skip_connection = (
                    nn.Linear(in_channels, out_channels)
                    if in_channels != out_channels
                    else nn.Identity()
                )

            layers.append(
                HGNNLayer(hgnn_layer, bn_layer, dropout_layer, skip_connection)
            )

        self.hgnn_layers = nn.Sequential(*layers)
        self.to(device)
        self.embedding_processor = cfg.model
        self.take_global_mean_pool_embedding = cfg.take_global_mean_pool_embedding

    def forward(self, x):
        batch_size = x["index"].shape[0]
        graph_data = x["graph_data"]

        node_features = self.node_embedding_layer(graph_data.x)

        if self.hgnn_initial_embedding_type in [
            "learnable_per_node",
            "learnable_per_node_value",
        ]:
            this_x = graph_data.x[:, 0].view(batch_size, -1)
            learned_embeddings = self.node_embedding_layer_learnable(this_x)
            node_features = node_features + learned_embeddings.reshape(
                -1, self.ip_encoding_size
            )

        hyperedge_index = graph_data.edge_index
        hyperedge_attr = (
            graph_data.edge_attr if hasattr(graph_data, "edge_attr") else None
        )

        for layer in self.hgnn_layers:
            if self.debug:
                # check if the max value in hyperedge_index[1] is the same as the length of the node_features
                assert (
                    hyperedge_index[1].max().item() == hyperedge_attr.shape[0] - 1
                ), "Hyperedge index is not correct."

            node_features = layer(node_features, hyperedge_index, hyperedge_attr)

        self.process_embedding(x, batch_size, graph_data, node_features)
        return x

    def process_embedding(self, x, batch_size, graph_data, node_features):
        if self.embedding_processor == "nn" and self.take_global_mean_pool_embedding:
            graph_embeddings = global_mean_pool(
                node_features, graph_data.batch, size=batch_size
            )
            node_features_reshaped = graph_embeddings
        elif (
            self.embedding_processor == "transformer"
            or not self.take_global_mean_pool_embedding
        ):
            nodes_per_graph = node_features.shape[0] // batch_size
            node_features_reshaped = node_features.reshape(
                batch_size, nodes_per_graph, -1
            )  # [batch_size, nodes_per_graph, embedding_size]

        if self.embedding_processor == "transformer":
            graph_embeddings = node_features_reshaped
        elif self.embedding_processor == "nn":
            graph_embeddings = node_features_reshaped.reshape(batch_size, -1)
        x["embedding"] = graph_embeddings

    def _get_hgnn_layer(self, cfg, in_channels, out_channels):
        common_kwargs = {
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
            "bias": True,
        }

        hypergraph_class = self.HYPERGRAPH_CLASSES.get(cfg.hypergraph_class)
        if hypergraph_class is None:
            raise ValueError(f"Invalid hypergraph class: {cfg.hypergraph_class}")

        if cfg.hypergraph_class == "HypergraphConv":
            return hypergraph_class(
                **common_kwargs,
                use_attention=True,
                attention_mode=cfg.hgnn_attention_mode,
                heads=cfg.hgnn_heads,
                dropout=0 if cfg.no_dropout else cfg.dropout_rate,
            )

        return hypergraph_class(
            **common_kwargs,
            drop_rate=0 if cfg.no_dropout else cfg.dropout_rate,
            use_bn=not cfg.no_batchnorm,
        )
