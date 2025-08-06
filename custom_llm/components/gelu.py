import torch
import math

class GeLU(torch.nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, input_vector):
        return (0.5 * input_vector) * (1 + math.tanh(math.sqrt(2/math.pi))) * (input_vector + 0.044715 * torch.pow(input_vector, 3))
