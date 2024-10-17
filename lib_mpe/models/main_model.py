import torch.nn as nn


class CombinedModel(nn.Module):
    def __init__(self, embedding_layer, main_model):
        """
        Initialize the CombinedModel class. Used to combine the embedding layer and the main model.

        Parameters:
        - embedding_layer: Embedding layer.
        - main_model: Main model.
        """
        super(CombinedModel, self).__init__()
        self.embedding_layer = embedding_layer
        self.main_model = main_model

    def get_embedding(self):
        """
        Get the embedding layer.

        Returns:
        - embedding_layer: Embedding layer.
        """
        if self.embedding_layer is not None:
            return self.embedding_layer

    def get_main_model(self):
        """
        Get the main model.

        Returns:
        - main_model: Main model.
        """
        return self.main_model
