import torch.nn as nn
import torch.nn.functional as F
from fbm_dropout.fbm_dropout import DropoutFBM

class DenseNet(nn.Module):

    def __init__(self, hidden_sizes, dropout_rates, device=None, dtype=None):

        super().__init__()
        
        self.linear_1 = nn.Linear(784, hidden_sizes[0], device=device, dtype=dtype)
        self.linear_2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], device=device, dtype=dtype)
        self.linear_3 = nn.Linear(hidden_sizes[1], 10, device=device, dtype=dtype)
        self.dropout_1 = nn.Dropout(dropout_rates[0])
        self.dropout_2 = nn.Dropout(dropout_rates[1])

    def forward(self, input):

        output = F.relu(self.dropout_1(self.linear_1(input)))
        output = F.relu(self.dropout_2(self.linear_2(output)))
        output = self.linear_3(output)
        return output

class DenseNetFBM(nn.Module):

    def __init__(self, hidden_sizes, n_agents, n_samples, max_iters, t_scale, grid_sizes, device=None, dtype=None):

        super().__init__()
        
        self.linear_1 = nn.Linear(28*28, hidden_sizes[0], device=device, dtype=dtype)
        self.linear_2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], device=device, dtype=dtype)
        self.linear_3 = nn.Linear(hidden_sizes[1], 10, device=device, dtype=dtype)
        self.dropout_1 = DropoutFBM(0.9, n_agents[0], n_samples, max_iters, t_scale, grid_sizes[0], device=device, dtype=dtype)
        self.dropout_2 = DropoutFBM(0.9, n_agents[1], n_samples, max_iters, t_scale, grid_sizes[1], device=device, dtype=dtype)

    def forward(self, input):

        if self.training:
            output = F.relu(self.dropout_1(self.linear_1(input)))
            output = F.relu(self.dropout_2(self.linear_2(output)))
        else:
            output = F.relu(self.linear_1(input))
            output = F.relu(self.linear_2(output))
            
        output = self.linear_3(output)
        
        return output