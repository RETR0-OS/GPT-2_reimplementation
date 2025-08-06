import torch
from .gelu import GeLU

class FeedForwardNN(torch.nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 3*embedding_dim),
            GeLU(),
            torch.nn.Linear(3*embedding_dim, embedding_dim)
        )

    def forward(self, input_vector):
        return self.layers(input_vector)