import torch

class CustomMultiHeadAttention(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim, context_len, num_heads=2, dropout=0.2, kqv_bias=False):

        super().__init__()
        assert (output_dim % num_heads) == 0, "Output dimension must be divisible by number of heads"

        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.w_q = torch.nn.Linear(input_dim, output_dim, bias=kqv_bias)
        self.w_k = torch.nn.Linear(input_dim, output_dim, bias=kqv_bias)
        self.w_v = torch.nn.Linear(input_dim, output_dim, bias=kqv_bias)

        self.out_proj = torch.nn.Linear(output_dim, output_dim, bias=kqv_bias)

        self.dropout = torch.nn.Dropout(dropout)

        self.register_buffer("attention_mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

    
    def forward(self, input_vector):

        batch_size, seq_len, input_dim = input_vector.shape

        q = self.w_q(input_vector)
        k = self.w_k(input_vector)
        v = self.w_v(input_vector)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)  # (batch_size, num_heads, head_dim, seq_len)
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        attn_scores = q @ k.transpose(2, 3)


        attn_scores = attn_scores.masked_fill(self.attention_mask.bool()[:seq_len, :seq_len], -torch.inf)

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ v  # (batch_size, num_heads, seq_len, head_dim)

        context_vector = context_vector.contiguous().view(batch_size, seq_len, self.output_dim)
        context_vector = self.out_proj(context_vector)

        return context_vector
    
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

    attn_block = CustomMultiHeadAttention(input_vector.shape[-1], 2, input_vector.shape[0], num_heads=2, dropout=0.2)
    print(attn_block(batch))