import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    def plot_state_values(self):
        plt.figure(figsize=(8,6))
        sns.heatmap(self.state_value_f, annot=True, cmap='coolwarm_r', fmt='.2f')
        plt.title('State value function')
        plt.axis(emit=True)
        plt.show()


if __name__ == "__main__":
    policy = {}
    world = GridWorld(4)
    #random policy
    for x in range(world.grid_size):
            for y in range (world.grid_size):
                policy[(x,y)] = world.actions.keys()   
    
    policy_evaluation = PolicyEvaluation(policy, world)
    policy_evaluation.iterative_policy_evaluation()
    policy_evaluation.plot_state_values()



