import torch

class LayerNomalization(torch.nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(embedding_dim))
        self.shift = torch.nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, input_vector):
        mean = input_vector.mean(dim=-1, keepdims=True)
        variance = input_vector.var(dim=-1, keepdims=True, unbiased=True)

        normalized_tensor = (input_vector - mean) / torch.sqrt(variance + self.eps)

        return self.scale * normalized_tensor + self.shift


