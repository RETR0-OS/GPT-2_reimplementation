import torch

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, vocab_size:int, embedding_dim: int=256):

        super().__init__()
        
        self.embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, input_batch):
        
        batch_results = []
        for i in range(input_batch.shape[0]):
            q_results = []
            for token in input_batch[i]:
                embeddings = self.embedding_layer(token)
                q_results.append(embeddings)
            batch_results.append(torch.stack(q_results, dim=0))

        return torch.stack(batch_results, dim=0)  # Stack the results to maintain batch dimension

