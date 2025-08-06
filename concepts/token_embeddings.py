from torch.nn import Embedding

# Scaled down implementation: 10 vocabulary size, 5 embedding dimensions
vocab_size = 10
embedding_dim = 5

#Embedding Matrix dims = vocab_size x embedding_dim
embedding_matrix = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# The embedding matrix performs the same operation as the Linear layer. 
# The only reason for a separate embeding laye is because embedding laye is more efficient.

print(f"Embedding matrix created with dimensions: {embedding_matrix.weight.shape}")
print(f"Embedding matrix weights\n: {list(embedding_matrix.weight)}")