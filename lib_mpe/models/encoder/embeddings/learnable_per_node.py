import torch
import torch.nn as nn


class LearnablePerNodeEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim, device):
        super(LearnablePerNodeEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.device = device

        # Create embedding layer for nodes
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)

        # Initialize embeddings
        self.initialize_embeddings()

    def initialize_embeddings(self):
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, node_values):
        batch_size = node_values.shape[0]
        # Create indices tensor for the batch
        node_indices = torch.arange(self.num_nodes, device=self.device)

        # Get embeddings for each node
        embeddings = self.embeddings(node_indices)

        # Expand embeddings to match batch size
        batch_embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        return batch_embeddings
