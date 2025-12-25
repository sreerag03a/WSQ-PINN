import torch
import torch.nn as nn


'''


'''

class PINN_WSQ(nn.Module):

    def __init__(self,activation_function = nn.Tanh,dropout = 0.2):
        super().__init__()
        self.net =  nn.Sequential(
            nn.Linear(4,128),activation_function(), # First layer with input dimension (5,) and tanh activation function
            nn.Linear(128,128), activation_function(), # Second layer with 128 inputs and tanh activation function
            nn.Linear(128,128), activation_function(),
            nn.Linear(128,3) # Outputs H(z), x(z), y(z)
        )

    def forward(self,x):
        return self.net(x)
    