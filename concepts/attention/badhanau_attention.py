import torch

# A smaple embedding vector with 3 dimensions for illustration

input_vector = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55],
    ])

words = ["Your", "journey", "starts", "with", "one", "step"]

"""
To find the attention scores, we calculate the dot product of the query vector with every other input vectors.
"""

## Focussing on query vectory as "journey"
# query_vector = input_vector[1]  # "journey" vector
# attention_scores_q2 = torch.empty(input_vector.shape[0])

# for i, x_i in enumerate(input_vector):
#     attention_scores_q2[i] = torch.dot(query_vector, x_i) #dot product

# print("Attention scores for 'journey':", attention_scores_q2)

# # normalizing the attention scores for training stability
# attention_scores_q2 = attention_scores_q2 / attention_scores_q2.sum()
# print("Normalized attention weights:", attention_scores_q2)

# # better normalization using softmax
# ## Pytorch softmax function: e^(x_i - max(x)) / sum(e^(x_i - max(x)))

# def naive_softmax(x):
#     return torch.exp(x) / torch.exp(x).sum(dim=0)

# attention_weights_q2_softmax = torch.softmax(attention_scores_q2, dim=0)
# print("naive softmax attention scores:", naive_softmax(attention_scores_q2))
# print("Softmax normalized attention weights:", attention_weights_q2_softmax)
# print("Sum of attention weights:", attention_weights_q2_softmax.sum())

# # Calculate context vectors for "journey"
# context_vector_q2 = torch.zeros(input_vector.shape[1])

# for i, x_i in enumerate(input_vector):
#     context_vector_q2 += attention_weights_q2_softmax[i] * x_i  # weighted sum
# print("Context vector for 'journey':", context_vector_q2)

def get_attention_scores(input_vectors):
    attention_scores = input_vectors @ input_vectors.T  # Dot product with all vectors
    # Normalize the attention scores for stability
    # Using softmax to ensure the scores sum to 1
    return attention_scores
    
def get_context_vector(input_vectors):
    
    attention_scores = get_attention_scores(input_vectors)

    attention_weights = torch.softmax(attention_scores, dim=-1)

    context_vector = attention_weights.matmul(input_vectors)
    
    return context_vector

def print_attention_weight_matrix(input_vectors):

    print("Attention Matrix:", end="\t\n")
    print("\t", end="")
    print(words)
    
    attention_scores = torch.softmax(get_attention_scores(input_vectors), dim = -1)
    print(attention_scores)

print_attention_weight_matrix(input_vector)
print("Context Vector for all words:")
print(get_context_vector(input_vector))
