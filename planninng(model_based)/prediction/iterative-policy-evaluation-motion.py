import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

class GridWorld:
    actions = { 
        1:(1,0),  #up
        2:(-1,0), #down
        3:(0,1),  #right
        4:(0,-1)  #left
    }

    def __init__(self, grid_size):
        self.grid_size= grid_size

    def is_terminal(self, state: tuple):
        return state in [(0,0), (self.grid_size-1,self.grid_size-1)]

    def step(self, state: tuple, action: int):
        if self.is_terminal(state):
            return state, 0
        
        dx, dy = self.actions[action]
        next_x = np.clip(state[0] + dx, 0, self.grid_size - 1)
        next_y = np.clip(state[1] + dy, 0, self.grid_size - 1)

        if self.is_terminal((next_x, next_y)):
            return (next_x, next_y), 0
        else:
            return (next_x, next_y), -1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

class GridWorld:
    actions = { 
        1:(1,0),  #up
        2:(-1,0), #down
        3:(0,1),  #right
        4:(0,-1)  #left
    }

    def __init__(self, grid_size):
        self.grid_size= grid_size

    def is_terminal(self, state: tuple):
        return state in [(0,0), (self.grid_size-1,self.grid_size-1)]

    def step(self, state: tuple, action: int):
        if self.is_terminal(state):
            return state, 0
        
        dx, dy = self.actions[action]
        next_x = np.clip(state[0] + dx, 0, self.grid_size - 1)
        next_y = np.clip(state[1] + dy, 0, self.grid_size - 1)

        if self.is_terminal((next_x, next_y)):
            return (next_x, next_y), 0
        else:
            return (next_x, next_y), -1
