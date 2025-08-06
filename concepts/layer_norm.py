import torch

class CustomLayerNorm(torch.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(embedding_dim))
        self.shift = torch.nn.Parameter(torch.zeros(embedding_dim))
    
    def forward(self, input_tensor):
        mean = input_tensor.mean(dim=-1, keepdim=True)
        variance = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        normalized_tensor = (input_tensor - mean) / torch.sqrt(variance + self.eps)

        return self.scale * normalized_tensor + self.shift


batch_example = torch.randn(2, 5)
layer = torch.nn.Sequential(
    torch.nn.Linear(5, 5),
    torch.nn.ReLU()
)
out = layer(batch_example)

print("Input batch shape:", batch_example.shape)
print("Output shape:", out.shape)
print("Output tensor:\n", out)

layer_norm = CustomLayerNorm(embedding_dim=5)
output = layer_norm(out)
print("Layer normalized output shape:", output.shape)
print("Layer normalized output tensor:\n", output)  