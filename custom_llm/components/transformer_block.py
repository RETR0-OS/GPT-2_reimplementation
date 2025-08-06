import torch
from .layer_normalizing import LayerNomalization
from .multi_head_attention import MultiHeadAttention
from .feedforward_network import FeedForwardNN

class TransformerBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out, context_len, num_heads, dropout=0.1, kqv_bias=False, skip_conn=True):

        super().__init__()
        
        self.layer_normalization_1 = LayerNomalization(dim_in)
        self.attention_block = MultiHeadAttention(dim_in, dim_out, context_len, num_heads, dropout, kqv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.feedforward = FeedForwardNN(dim_in)
        self.layer_normalization_2 = LayerNomalization(dim_in)

        self.skip_connections = skip_conn

    def forward(self, input_vector):

        layer_out = self.layer_normalization_1(input_vector)
        layer_out = self.attention_block(layer_out)
        layer_out = self.dropout(layer_out)

        if self.skip_connections:
            layer_out += input_vector
            intermediate_out = layer_out

        layer_out = self.layer_normalization_2(layer_out)
        layer_out = self.feedforward(layer_out)
        layer_out = self.dropout(layer_out)

        if self.skip_connections:
            layer_out += intermediate_out
        
        return layer_out