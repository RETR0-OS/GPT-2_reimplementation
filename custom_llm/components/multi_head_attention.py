import torch

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.1, kqv_bias=False):

        super().__init__()

        assert d_out % num_heads == 0, "Output dimension must be divisible by the number of heads"

        self.output_dim = d_out

        self.w_q = torch.nn.Linear(d_in, d_out, bias=kqv_bias)
        self.w_k = torch.nn.Linear(d_in, d_out, bias=kqv_bias)
        self.w_v = torch.nn.Linear(d_in, d_out, bias=kqv_bias)

        self.head_dim = d_out // num_heads
        self.num_heads = num_heads

        self.dropout = torch.nn.Dropout(dropout)

        self.out_proj = torch.nn.Linear(d_in, d_out, bias=kqv_bias)

        self.register_buffer("attention_mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, input_batch):
        
        batch_size, ctx_len, embed_dims = input_batch.shape
        
        k = self.w_k(input_batch)
        q = self.w_q(input_batch)
        v = self.w_v(input_batch)

        k = k.view(batch_size, ctx_len, self.num_heads, self.head_dim) # Reshape to (batch_size, ctx_len, num_heads, head_dim) to split into heads
        q = q.view(batch_size, ctx_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, ctx_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2) # group by heads for easier matrix multiplication
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        attention_scores = q @ k.transpose(2, 3) # calculate attention scores (batch_size, num_heads, ctx_len, ctx_len)

        masked_attention_scores = torch.masked_fill(attention_scores, self.attention_mask.bool()[:ctx_len, :ctx_len], -torch.inf) # apply attention mask to prevent attending to future tokens

        attention_weights = torch.softmax(masked_attention_scores / (self.head_dim ** 0.5), dim=-1) #Apply softmax along columns to add up rows to 1

        attention_weights = self.dropout(attention_weights) # apply dropout to attention weights
        
        context_vector = attention_weights @ v # (batch_size, num_heads, ctx_len, head_dim)

        context_vector = context_vector.contiguous().view(batch_size, ctx_len, self.output_dim) # Reshape back to (batch_size, ctx_len, output_dim) for output compatibility
        context_vector = self.out_proj(context_vector) # apply linear projection to combine heads

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

    mha = MultiHeadAttention(input_vector.shape[-1], input_vector.shape[-1], input_vector.shape[0], num_heads=3, kqv_bias=True)
    print(mha(batch))

