import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

class Visualize:
    def __init__(self, world,state_value_f, policy):
        self.world = world
        self.state_value_f= state_value_f
        self.policy = policy

    def plot_state_values(self):
        plt.figure(figsize=(8,6))
        sns.heatmap(self.state_value_f, annot=True, cmap='coolwarm_r', fmt='.2f')
        plt.title('State value function')
        plt.axis(emit=True)
        plt.show()
    
    def plot_policy(self):
        plt.figure(figsize=(8,8))
        for row in range(self.world.grid_size):
            for col in range(self.world.grid_size):
                if self.world.is_terminal((row, col)):
                    continue
                plt.text(col, row, '⬆⬇➡⬅', ha='center', va='center')
        plt.title('Policy visualization (Uniform random policy)')
        plt.xlim(-0.5, self.world.grid_size - 0.5)
        plt.ylim(-0.5, self.world.grid_size - 0.5)
        plt.gca().invert_yaxis()
        plt.grid()
        plt.show()

# define possible actions
class GridWorld:
    actions = { 
        1:(1,0),  #up
        2:(-1,0), #down
        3:(0,1),  #dx
        4:(0,-1) #sx
    }
    
    def __init__(self,grid_size):
        self.grid_size= 4
    

    def is_terminal(self, state: tuple):
        if state in [(0,0),(self.grid_size-1,self.grid_size-1)]:
            return True
        else:
            return False
    
    # returns (next_state, return, done)
    # next_state
    # return:   0 if end is reached, -1 else
    def step(self, state: tuple, action: int):
        #if in terminal state, no more action to take and reward = 0
        if self.is_terminal(state):
            return state, 0
        
        #if not in terminal state, action to take
        dx,dy = self.actions[action]
        next_x = np.clip(state[0]+dx, 0,self.grid_size-1)
        next_y = np.clip(state[1]+dy, 0,self.grid_size-1)

        if self.is_terminal((next_x,next_y)):
            return (next_x,next_y), 0
        else:
            return (next_x,next_y), -1

class PolicyEvaluation:
    def __init__(self, policy: dict, world: GridWorld):
        self.state_value_f = np.zeros((world.grid_size,world.grid_size))
        self.new_state_value_f= None
        self.policy = policy
        self.world=world

        self.convergence = False
    
    def state_value_f_convergence(self):
        if (self.state_value_f-self.new_state_value_f).sum()>0.5:
            self.convergence = False
        else:
            self.convergence = True
    
    def iterative_policy_evaluation(self):
        while(self.convergence==False):
            # deep backup
            self.new_state_value_f = self.state_value_f.copy()

            for row in range(self.world.grid_size):
                for col in range(self.world.grid_size):
                    # for each state
                    state = (row,col)
                    state_value = 0

                    for a in self.world.actions.keys():
                        next_state, reward = self.world.step(state,a)
                        policy = 1 / len(self.policy[state])
                        state_value +=  policy*(reward+ self.state_value_f[next_state[0],next_state[1]])
                    
                    self.new_state_value_f[row,col] = state_value
            
            self.state_value_f_convergence()
            self.state_value_f = self.new_state_value_f
        return self.state_value_f


class PolicyIteration:
    def __init__(self, world: GridWorld):
        self.world = world
        
        # initial random policy
        self.policy = {}
        for x in range(world.grid_size):
            for y in range (world.grid_size):
                self.policy[(x,y)] = world.actions.keys()
        
        # initial state value function
        self.best_value_f_given_policy = np.zeros((self.world.grid_size, self.world.grid_size))
        self.new_best_value_f_given_policy = None
        self.convergence = False
    
    def state_value_f_convergence(self):
        if (self.best_value_f_given_policy-self.new_best_value_f_given_policy).sum()>0.5:
            self.convergence = False
        else:
            self.convergence = True
        
    def policy_impovement(self):
        for x in range(self.world.grid_size):
            for y in range(self.world.grid_size):
                # find best state to visitate after
                best_action = None
                best_value = float('-inf')

                for action in self.world.actions:
                    next_state, reward = self.world.step((x,y),action)
                    value = reward + self.best_value_f_given_policy[next_state]

                    if value > best_value:
                        best_value = value
                        best_action = action
                
                # check if policy is congruent, eventually update it
                if self.policy[(x,y)] != best_action:
                    self.policy[(x,y)] = [best_action]
                


    def policy_iteration(self):
        while(True):
            policy_evaluation = PolicyEvaluation (self.policy, self.world)
            self.new_best_value_f_given_policy = policy_evaluation.iterative_policy_evaluation()
            self.state_value_f_convergence()

            if self.convergence:
                break
            else:
                self.best_value_f_given_policy = self.new_best_value_f_given_policy
            
            # improve policy
            self.policy_impovement()
        
        #convergence is reached
        visualizer = Visualize(self.world, self.best_value_f_given_policy, self.policy)
        visualizer.plot_state_values()
        visualizer.plot_policy()



if __name__ == "__main__":
    policy_iteration = PolicyIteration(GridWorld(4))
    policy_iteration.policy_iteration()


    
