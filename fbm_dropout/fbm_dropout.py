import torch
from stochastic.processes.continuous import FractionalBrownianMotion
import torch.nn as nn
from torch import Tensor
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class fbm_dropout(nn.Module):

    def __init__(self, hurst: float, n_fibers: int, max_epoch: int, input_size: int, show=False):

        super().__init__()
        
        self.fibers_x = []
        self.fibers_y = []
        self.colors = []
        self.n_fibers = n_fibers
        self.input_size = input_size
        self.show = show
        for _ in range(n_fibers):
            fbm_x = FractionalBrownianMotion(hurst, t=1)
            fbm_y = FractionalBrownianMotion(hurst, t=1)

            self.fibers_x.append((fbm_x.sample(100 * max_epoch) + torch.rand((1,)).item()) % 1)
            self.fibers_y.append((fbm_y.sample(100 * max_epoch) + torch.rand((1,)).item()) % 1)
            color = torch.rand((3,))
            color = (color[0].item(), color[1].item(), color[2].item())
            self.colors.append(color)
        
        self.grid = self.get_grid()

    def forward(self, input: Tensor, current_epoch) -> Tensor:
        '''
        given an input, return a dropped out output
        where the dropout probability of each neuron
        is determined by the fbm

        args:
            input: a Tensor with shape (batch_size, self.input_size)

        '''
        mask = self.get_mask(current_epoch)

        return torch.mul(input, mask)

    def get_mask(self, epoch: int) -> Tensor:

        self.mask = self.is_touching(epoch)

        return self.mask

    def get_grid(self):

        n_row = n_col = math.ceil(math.sqrt(self.input_size))

        size_row = 1.0 / n_row
        size_col = 1.0 / n_col

        gap_y = size_row / 4
        gap_x = size_col / 4

        grid = {}
        for i in range(self.input_size):
            y = i // n_row
            x = i % n_col

            x_low = (x * size_col) + gap_x
            y_low = (y * size_row) + gap_y
            x_high = x_low + 2 * gap_x
            y_high = y_low + 2 * gap_y
            grid[i] = ((x_low, x_high), (y_low, y_high))

        return grid

    def is_touching(self, epoch: int):
        
        def is_in(fiber_x, fiber_y, g):
            ((x_low, x_high), (y_low, y_high)) = g
            for x,y  in zip(fiber_x, fiber_y):
                if x_low <= x and x <= x_high and y_low <= y and y <= y_high:
                    return True
            return False

        is_touching = torch.ones((self.input_size,))

        t = epoch*100
        curr_fiber_x = [fiber_x[t:t+100] for fiber_x in self.fibers_x]
        curr_fiber_y = [fiber_y[t:t+100] for fiber_y in self.fibers_y]

        if self.show:
            fig, ax = plt.subplots()
            for i in range(self.n_fibers):
                ax.scatter(curr_fiber_x[i], curr_fiber_y[i], s=1.0, color=self.colors[i])
            for i in range(self.input_size):
                ((x_low, x_high), (y_low, y_high)) = self.grid[i]
                ax.add_patch(Rectangle((x_low, y_low), x_high - x_low, y_high - y_low,
                                       facecolor='red', zorder=0))
            ax.set_aspect('equal')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.show()
        
        for i in range(self.input_size):
            for x, y in zip(curr_fiber_x, curr_fiber_y):
                if is_in(x, y, self.grid[i]):
                    is_touching[i] = 0

        return is_touching
    
    def get_dropout_rate(self):
        return 1 - (torch.sum(self.mask) / self.input_size)

class DropoutFBM(nn.Module):

    def __init__(self, hurst: float, n_agents: int, max_epoch: int, grid_size: tuple, is_conv=False, show=False, device=None, dtype=None):

        super().__init__()
        
        # independent FBM's for x and y coordinates
        self.agents_x = []
        self.agents_y = []
        # colors for each FBM droput agents
        self.colors = []
        # the number of FBM dropout agents = len(self.agents_x) = len(self.agents.y) = len(colors)
        self.n_agents = n_agents
        self.grid_size = grid_size
        # true if the dropout is after conv layer and false if after linear layer
        self.is_conv = is_conv
        # true if want to print out the dropout image
        self.show = show
        
        self.cum_dropout_rate = 0.0
        self.curr_dropout_rate = 0.0
        
        self.dtype = dtype
        self.device = device

        # keeps track of its current epoch
        self.current_epoch = -1

        # initialize (x, y) of FBM dropout agents for max epochs
        for _ in range(n_agents):
            fbm_x = FractionalBrownianMotion(hurst, t=1)
            fbm_y = FractionalBrownianMotion(hurst, t=1)

            self.agents_x.append((fbm_x.sample(100 * max_epoch) + torch.rand((1,)).item()) % 1)
            self.agents_y.append((fbm_y.sample(100 * max_epoch) + torch.rand((1,)).item()) % 1)
            color = torch.rand((3,))
            color = (color[0].item(), color[1].item(), color[2].item())
            self.colors.append(color)
        
        # initialize the grid of neurons
        self.grid = self.get_grid()

    def forward(self, input: Tensor, current_epoch) -> Tensor:
        '''
        given an input, return a dropped out output
        where the dropout probability of each neuron
        is determined by the fbm

        args:
            input: a Tensor with shape (batch_size, self.input_size)

        '''
        # mask stays constant for each epoch
        if self.current_epoch != current_epoch:
            # get mask
            self.current_epoch = current_epoch
            self.get_mask(current_epoch)
            # add dropout rate
            self.cum_dropout_rate += self.get_dropout_rate()
            self.curr_dropout_rate = self.cum_dropout_rate / (current_epoch + 1)
            # move to dtype and device
            if self.dtype:
                self.mask = self.mask.to(self.dtype)
            if self.device:
                self.mask = self.mask.to(self.device)

        # return input * mask
        return torch.mul(input, self.mask)

    def get_mask(self, epoch: int):

        self.mask = self.is_touching(epoch)

    def get_grid(self):

        self.n_row = n_row = self.grid_size[1]
        self.n_col = n_col = self.grid_size[0]

        # get size of rectangle for each neuron
        size_row = 1.0 / n_row
        size_col = 1.0 / n_col

        # partition the rectangle into 16 
        gap_y = size_row / 4
        gap_x = size_col / 4

        # for each neuron, out of 16, 4 in the middle are for neuron
        # and 12 on the boundary are for blank space
        grid = {}
        for x in range(n_col):
            for y in range(n_row):
                x_low = (x * size_col) + gap_x
                y_low = (y * size_row) + gap_y
                x_high = x_low + 2 * gap_x
                y_high = y_low + 2 * gap_y
                grid[(x,y)] = ((x_low, x_high), (y_low, y_high))

        return grid

    def is_touching(self, epoch: int):
        
        def is_in(agent_x, agent_y, g):
            # bounding box for a neuron
            ((x_low, x_high), (y_low, y_high)) = g
            # check if (x, y) is within the bounding box 
            for x,y  in zip(agent_x, agent_y):
                if x_low <= x and x <= x_high and y_low <= y and y <= y_high:
                    return True
            return False

        # initialize a mask
        is_touching = torch.ones(self.grid_size)

        # get (x, y) for current epoch
        t = epoch*100
        curr_agent_x = [agent_x[t:t+100] for agent_x in self.agents_x]
        curr_agent_y = [agent_y[t:t+100] for agent_y in self.agents_y]
        
        # set is_touching[(i,j)] to 0 if (x,y) is within neuron (i,j)
        for i in range(self.n_col):
            for j in range(self.n_row):
                for x, y in zip(curr_agent_x, curr_agent_y):
                    if is_in(x, y, self.grid[(i,j)]):
                        is_touching[(i,j)] = 0
        
        # print the current grid status
        if self.show:
            fig, ax = plt.subplots(figsize=(10,10))
            # print the agents
            for i in range(self.n_agents):
                ax.scatter(curr_agent_x[i], curr_agent_y[i], s=1.0, color=self.colors[i])
            # print the neurons
            for x in range(self.n_col):
                for y in range(self.n_row):
                    ((x_low, x_high), (y_low, y_high)) = self.grid[(x,y)]
                    if is_touching[(x,y)]:
                        # set to black if not dropped out
                        facecolor = 'black'
                    else:
                        # set to red if dropped out
                        facecolor = 'red'
                    ax.add_patch(Rectangle((x_low, y_low), x_high - x_low, y_high - y_low,
                                           facecolor=facecolor, zorder=0))
            ax.set_aspect('equal')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Dropout rate: {:2f}'.format((1 - (torch.sum(is_touching) / (self.n_col * self.n_row)).item())))
            plt.show()
        
        # rotate 90 degrees
        is_touching = torch.rot90(is_touching)
        if self.is_conv:
            return is_touching
        # flatten for linear layer
        return is_touching.reshape(-1)

    def get_dropout_rate(self):

        if self.current_epoch > -1:
            return (1 - (torch.sum(self.mask) / (self.n_col * self.n_row))).item()
        else:
            return None