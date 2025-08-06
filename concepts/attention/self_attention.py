import torch

"""
Todo: Convert the incoming input_embeddings vector into k, q, v vectors using the self attention mechanism
"""

class CustomSelfAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kqv_bias=False):
        super().__init__()
        self.w_q = torch.nn.Linear(input_dim, output_dim, bias=kqv_bias) #nn.linear provides a more optimized weight initialization scheme than nn.Parameter.  
        self.w_k = torch.nn.Linear(input_dim, output_dim, bias=kqv_bias) #nn.linear provides a more optimized weight initialization scheme than nn.Parameter. 
        self.w_v = torch.nn.Linear(input_dim, output_dim, bias=kqv_bias) #nn.linear provides a more optimized weight initialization scheme than nn.Parameter. 

    def forward(self, input_vector):
        q = input_vector @ self.w_q
        k = input_vector @ self.w_k
        v = input_vector @ self.w_v

        attention_scores = q @ k.T
        d_k = k.shape[-1]
        attention_weights = torch.softmax(attention_scores / (d_k ** 0.5), dim=-1)
        context_vector = attention_weights @ v

        return context_vector

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

if __name__ == "__main__":
    attention_block = CustomSelfAttention(input_vector.shape[1], input_vector.shape[1])

    context_vector = attention_block(input_vector)
    print("Context vector:", context_vector)
