import numpy as np
def sampling_next_state(states: np.array, p_matrix: np.array ,curr_state: str):
    # cumulative probability implementation

    #1 select transition probability given current state
    index_row_p_matrix = np.where(states==curr_state)[0][0]
    transition_probability_from_current_state = p_matrix[index_row_p_matrix,:]

    #2 generate random number
    random_number=np.random.rand()

    #3 look for next state
    cumulative_prob=0
    state_position=-1
    for next_state_prob in transition_probability_from_current_state:
        cumulative_prob+=next_state_prob
        state_position+=1

        if random_number<cumulative_prob:
            return states[state_position]




states=np.array(['c1','c2','c3','Pass','Pub','TK','Sleep'])
p_matrix=np.array([ [0.0,   0.5,    0.0,    0.0,    0.0,    0.5,    0.0],
                    [0.0,   0.0,    0.8,    0.0,    0.0,    0.0,    0.2],
                    [0.0,   0.0,    0.0,    0.6,    0.4,    0.0,    0.0],
                    [0.0,   0.0,    0.0,    0.0,    0.0,    0.0,    1.0],
                    [0.2,   0.4,    0.4,    0.0,    0.0,    0.0,    0.0],
                    [0.1,   0.0,    0.0,    0.0,    0.0,    0.9,    0.0],
                    [0.0,   0.0,    0.0,    0.0,    0.0,    0.0,    0.1]
                    ]
                  )

value_function=np.zeros((7))
rewards=np.array([-2,   -2, -2, 10, 1,  -1, 0])

gamma=0.9

for init_state in states:
    for num_episode in range(10000):
        #create an episode with initial statse==init_state
        e_return=0
        curr_state=init_state
        timestamp=0

        while curr_state != 'Sleep':
            # update current state
            e_return+=rewards[np.where(states==curr_state)[0][0]]*(gamma**timestamp)
            curr_state=sampling_next_state(states,p_matrix,curr_state)
            timestamp+=1
        
        value_function[np.where(states==init_state)[0][0]]+=e_return
    value_function[np.where(states==init_state)[0][0]]/=10000

print(value_function)
    
