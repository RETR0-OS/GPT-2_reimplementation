import torch
from GeLU import GeLU

class FeedForward(torch.nn.Module):

    def __init__(self, dim_in, kqv_bias=False):
        super().__init__()

        self.feedforward_block = torch.nn.Sequential(
            torch.nn.Linear(dim_in, 3*dim_in, bias=kqv_bias),
            GeLU(),
            torch.nn.Linear(3*dim_in, dim_in, bias=kqv_bias)
        )

    def forward(self, input_vector):

        return self.feedforward_block(input_vector)
    
if __name__ == "__main__":
    batch_example = torch.randn(2, 3, 768)
    feedforward_layer = FeedForward(dim_in=768)
    output = feedforward_layer(batch_example)

    print("Input batch shape:", batch_example.shape)
    print("Output shape:", output.shape)
    print("Output tensor:\n", output)