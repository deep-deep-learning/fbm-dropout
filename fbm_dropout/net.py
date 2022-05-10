import torch.nn as nn
import torch.nn.functional as F
from fbm_dropout.fbm_dropout import DropoutFBM_2, DropoutFBM_3, DropoutFBM_with_branching

class DenseNet(nn.Module):

    def __init__(self, hidden_sizes, dropout_rate, device=None, dtype=None):

        super().__init__()
        
        self.linear_1 = nn.Linear(784, hidden_sizes[0], device=device, dtype=dtype)
        self.linear_2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], device=device, dtype=dtype)
        self.linear_3 = nn.Linear(hidden_sizes[1], 10, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):

        output = F.relu(self.dropout(self.linear_1(input)))
        output = F.relu(self.dropout(self.linear_2(output)))
        output = self.linear_3(output)
        return output

class DenseNetFBM(nn.Module):

    def __init__(self, hidden_sizes, n_agents, grid_sizes, show=False, device=None, dtype=None):

        super().__init__()
        
        self.linear_1 = nn.Linear(28*28, hidden_sizes[0], device=device, dtype=dtype)
        self.linear_2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], device=device, dtype=dtype)
        self.linear_3 = nn.Linear(hidden_sizes[1], 10, device=device, dtype=dtype)
        self.dropout_1 = DropoutFBM_3(0.9, n_agents[0], grid_sizes[0], show=show, device=device, dtype=dtype)
        self.dropout_2 = DropoutFBM_3(0.9, n_agents[1], grid_sizes[1], show=show, device=device, dtype=dtype)

    def forward(self, input):

        if self.training:
            output = F.relu(self.dropout_1(self.linear_1(input)))
            output = F.relu(self.dropout_2(self.linear_2(output)))
        else:
            output = F.relu(self.linear_1(input))
            output = F.relu(self.linear_2(output))
            
        output = self.linear_3(output)
        
        return output

class DenseNetFBM_2(nn.Module):

    def __init__(self, hidden_sizes, n_agents, grid_sizes, max_iter, show=False, device=None, dtype=None):

        super().__init__()
        
        self.linear_1 = nn.Linear(28*28, hidden_sizes[0], device=device, dtype=dtype)
        self.linear_2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], device=device, dtype=dtype)
        self.linear_3 = nn.Linear(hidden_sizes[1], 10, device=device, dtype=dtype)
        self.dropout_1 = DropoutFBM_2(0.9, n_agents[0], max_iter, grid_sizes[0], show=show, device=device, dtype=dtype)
        self.dropout_2 = DropoutFBM_2(0.9, n_agents[1], max_iter, grid_sizes[1], show=show, device=device, dtype=dtype)

    def forward(self, input):

        if self.training:
            output = F.relu(self.dropout_1(self.linear_1(input)))
            output = F.relu(self.dropout_2(self.linear_2(output)))
        else:
            output = F.relu(self.linear_1(input))
            output = F.relu(self.linear_2(output))
            
        output = self.linear_3(output)
        
        return output

class DenseNetFBM_with_branching(nn.Module):

    def __init__(self, hidden_sizes, n_agents, grid_sizes, max_iter, show=False, device=None, dtype=None):

        super().__init__()
        
        self.linear_1 = nn.Linear(28*28, hidden_sizes[0], device=device, dtype=dtype)
        self.linear_2 = nn.Linear(hidden_sizes[0], hidden_sizes[1], device=device, dtype=dtype)
        self.linear_3 = nn.Linear(hidden_sizes[1], 10, device=device, dtype=dtype)
        self.dropout_1 = DropoutFBM_with_branching(0.9, n_agents[0], max_iter, grid_sizes[0], show=show, device=device, dtype=dtype)
        self.dropout_2 = DropoutFBM_with_branching(0.9, n_agents[1], max_iter, grid_sizes[1], show=show, device=device, dtype=dtype)

    def forward(self, input):

        if self.training:
            output = F.relu(self.dropout_1(self.linear_1(input)))
            output = F.relu(self.dropout_2(self.linear_2(output)))
        else:
            output = F.relu(self.linear_1(input))
            output = F.relu(self.linear_2(output))
            
        output = self.linear_3(output)
        
        return output