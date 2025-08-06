from causal_attention import CustomCausalAttention
import torch

class CustomMultiHeadAttentionWrapper(torch.nn.Module):

    def __init__(self, input_dim, output_dim, context_len, num_heads=2, dropout=0.2, kqv_bias=False):
        super().__init__()
        
        self.heads = torch.nn.ModuleList([
            CustomCausalAttention(input_dim, output_dim, context_len, dropout, kqv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, input_vector):
        return torch.cat([head(input_vector) for head in self.heads], dim=-1)


if __name__ == "__main__":
    
    input_vector = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55],
    ])

    batch = torch.stack((input_vector, input_vector), dim=0)

    words = ["Your", "journey", "starts", "with", "one", "step"]

    attn_block = CustomMultiHeadAttentionWrapper(input_vector.shape[-1], 2, input_vector.shape[0], num_heads=2, dropout=0.2)
    print(attn_block(batch))

