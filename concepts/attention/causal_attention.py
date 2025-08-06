import torch

class CustomCausalAttention(torch.nn.Module):

    def __init__(self, input_dim, output_dim, context_len, dropout=0.2, kqv_bias=False):
        super().__init__()
        #input_vector (6, 3)
        self.w_q = torch.nn.Linear(input_dim, output_dim, bias=kqv_bias) # (6, 2)
        self.w_k = torch.nn.Linear(input_dim, output_dim, bias=kqv_bias)
        self.w_v = torch.nn.Linear(input_dim, output_dim, bias=kqv_bias)

        self.register_buffer("attention_mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_vector):

        #input vector of shape (batch size, context_len, embedding_dims)

        q_proj = self.w_q(input_vector) # (6, 6)
        k_proj = self.w_k(input_vector)
        v_proj = self.w_v(input_vector)

        attention_scores = q_proj @ k_proj.transpose(1, 2)

        context_length = attention_scores.shape[1]
        
        masked_attention_scores = attention_scores.masked_fill(self.attention_mask.bool()[:context_length, :context_length], -torch.inf) # type: ignore

        attention_weights = masked_attention_scores / k_proj.shape[-1] ** 0.5

        attention_weights = torch.softmax(attention_weights, dim=-1)

        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ v_proj
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

    words = ["Your", "journey", "starts", "with", "one", "step"]

    attn_block = CustomCausalAttention(input_vector.shape[-1], 2, input_vector.shape[0], dropout=0.2)

    print(attn_block(torch.stack((input_vector, input_vector), dim=0)))