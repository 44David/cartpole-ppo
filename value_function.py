import torch 
import torch.nn as nn

class ValueFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(4, 64)
        self.hidden = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output_layer(x)
        
        return x
    
    
    
def value_target(reward_vector):
    gamma_discount = 0.99
    gamma_discount * reward_vector
    
    