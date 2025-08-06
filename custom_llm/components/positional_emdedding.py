import torch

class PositionalEmbedding(torch.nn.Module):

    def __init__(self, context_length, embedding_dims):

        super().__init__()

        self.positional_embedding = torch.nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_dims)

    def forward(self, token_embeddings):
        batch_size, seq_len, _ = token_embeddings.shape
        positions = torch.arange(seq_len, dtype=torch.long, device=token_embeddings.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len) # Expand to match batch size
        pos_emb = self.positional_embedding(positions)
        return token_embeddings + pos_emb # Return the input embeddings with positional embeddings added
