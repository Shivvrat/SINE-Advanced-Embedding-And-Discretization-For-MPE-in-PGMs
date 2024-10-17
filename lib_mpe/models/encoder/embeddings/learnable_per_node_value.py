import torch
import torch.nn as nn


class LearnablePerNodeValueEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim, device):
        super(LearnablePerNodeValueEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.device = device
        self.node_indices = torch.arange(num_nodes).to(device)

        # Create three separate embedding layers for each possible value
        self.embedding_neg = nn.Embedding(num_nodes, embedding_dim)
        self.embedding_zero = nn.Embedding(num_nodes, embedding_dim)
        self.embedding_pos = nn.Embedding(num_nodes, embedding_dim)

        # Initialize embeddings
        self.initialize_embeddings()

    def initialize_embeddings(self):
        nn.init.xavier_uniform_(self.embedding_neg.weight)
        nn.init.xavier_uniform_(self.embedding_zero.weight)
        nn.init.xavier_uniform_(self.embedding_pos.weight)

    def forward(self, node_values):
        # node_values: tensor of shape (batch_size, num_nodes)
        batch_size = node_values.shape[0]

        # Create node indices tensor for the batch
        node_indices = self.node_indices.expand(batch_size, -1)

        # Create masks for each value
        mask_neg = node_values == -1
        mask_zero = node_values == 0
        mask_pos = node_values == 1

        # Get embeddings for each value
        emb_neg = self.embedding_neg(node_indices)
        emb_zero = self.embedding_zero(node_indices)
        emb_pos = self.embedding_pos(node_indices)

        # Combine embeddings using masks
        result = (
            mask_neg.unsqueeze(-1) * emb_neg
            + mask_zero.unsqueeze(-1) * emb_zero
            + mask_pos.unsqueeze(-1) * emb_pos
        )

        return result
