import torch
import torch.nn as nn

class CartPoleAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.input_layer = nn.Linear(4, 64)
        self.hidden_1 = nn.Linear(64, 64) 
        self.hidden_2 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_1(x)
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x



        
        