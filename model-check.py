import torch
from torchsummary import summary
from torchviz import make_dot

from raac.model import CrackDetectionModel

model = CrackDetectionModel().cuda()  # if using GPU
summary(model, input_size=(3, 448, 448))  # replace with your input dimensions

# Create a model (replace this with your own model)
model1 = CrackDetectionModel()

# Example input tensor
x = torch.randn(1, 3, 448, 448)  # Example input tensor

# Move the input tensor to the same device as the model's parameters
device = next(model1.parameters()).device  # Get the device from the model's parameters
x = x.to(device)

# Forward pass
y = model1(x)

# Visualize the computation graph
make_dot(y, params=dict(list(model1.named_parameters()))).render("model_graph", format="png")
