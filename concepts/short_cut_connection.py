import torch
from GeLU import GeLU

class ExampleNeuralNetwork(torch.nn.Module):
    def __init__(self, layer_sizes, use_shortcut=True):
        super().__init__()

        self.use_shortcut = use_shortcut
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[0], layer_sizes[1]),GeLU()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[1], layer_sizes[2]),GeLU()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[2], layer_sizes[3]),GeLU()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[3], layer_sizes[4]),GeLU()),
            torch.nn.Sequential(torch.nn.Linear(layer_sizes[4], layer_sizes[5]),GeLU()),
        ])

    def forward(self, input_tensor):
        for layer in self.layers[:-1]:
            layer_output = layer(input_tensor)
            if self.use_shortcut:
                input_tensor = input_tensor + layer_output
            else:
                input_tensor = layer_output
        return input_tensor
    
if __name__ == "__main__":
    layer_sizes = [3, 3, 3, 3, 3, 1]

    batch_example = torch.tensor([[1., 0., -1.]])
    model = ExampleNeuralNetwork(layer_sizes, use_shortcut=True)
    output = model(batch_example)
    
    target = torch.tensor([[0.0]])

    loss = torch.nn.MSELoss()

    loss = loss(output, target)

    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            print(f"Parameter {name} has gradient mean of: {param.grad.abs().mean().item()}")

