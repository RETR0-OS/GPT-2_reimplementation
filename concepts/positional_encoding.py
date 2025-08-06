"""
Embedings for positional encoding in transformer models.

In this module, the token embedding for 50257 tokens and 256 dimensions is created.

The positional encoding will be dones in batches of 8 with 16 tokens per batch and 256 dimensions.
"""

import torch
from torch.nn import Embedding
from io_pairs import create_dataloader

vocab_size = 50257
embedding_dim = 256
batch_size = 8
context_length = 16

token_embedding_layer = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) #Weight size 50257 X 256
positional_embedding_layer = Embedding(num_embeddings=context_length, embedding_dim=embedding_dim) #Weight size 1024 X 256

f = open("C:/Users/aadit/Projects/LLM_from_scratch/data/the_verdict.txt", "r")
dataloader = create_dataloader(f.read(), batch_size=batch_size, max_len=context_length, stride=2, shuffle=False, drop_last=True, num_workers=0)
data_iter = iter(dataloader)
batch = next(data_iter)

token_embeddings = token_embedding_layer(batch[0])  # Shape: (batch_size, context_length, embedding_dim)
print("Token embeddings shape:", token_embeddings.shape)
positional_embeddings = positional_embedding_layer(torch.arange(context_length))  # Shape: (context_length, embedding_dim)
print("Positional embeddings shape:", positional_embeddings.shape)
input_embeddings = token_embeddings + positional_embeddings.unsqueeze(0)  # Shape: (batch_size, context_length, embedding_dim)
print("Input embeddings shape:", input_embeddings.shape)
