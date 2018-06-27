"""
A simple example for Reinforcement Learning using Q-Learning method.
An agent 'o' is on the leftmost position of a one-dimensional world, and the treasure is on the rightmost position.
This program vividly shows the process that the agent finds treasure by the strategy of Q-Learning method.
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)   # Setting seed with a fixed value make the program reproducible

# global settings
N_STATES = 6    # the length between the agent and the treasure in the one-dimensional world
ACTIONS = ['left', 'right'] # available actions
EPSILON = 0.9   # greedy policy, there is a EPSILON probability that the agent will choose most valuable action, and a (1-EPSILON) probability that it choose a random action
ALPHA = 0.1    # learning rate
GAMMA = 0.9     # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move
SLEEP_TIME = 2.0    # sleep time while finishing one episode

def build_q_table(states, actions):
    '''
    :param states: states set
    :param action: actions set
    :return: Q table
    '''
    table = pd.DataFrame(
        np.zeros((states, len(actions))), # the initial value ofQ table are all zeros
        columns = actions,  # the name of columns are same with actions
    )
    return table

def choose_action(state, q_table):
    '''
    :param state: current state, specificaly the position of the agent
    :param q_table: Q table
    :return: action of current state
    '''
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()): # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(state, action):
    '''
    :param state: current state
    :param action: current action
    :return: next state and reward
    '''
    if action == 'right':   # move right
        if state == N_STATES - 2:   # get treasure!
            state_ = 'terminal'
            reward = 1
        else:
            state_ = state + 1
            reward = 0
    else:   # move left
        reward = 0
        if state == 0:
            state_ = 0  # can not move out of ranage
        else:
            state_ = state - 1
    return state_, reward

def update_env(state, episode, step_counter):
    '''
    :param state: current state
    :param episode: current episode
    :param step_counter: the number of steps agent uses to reach current state in current episode
    :return:
    '''
    env_list = ['-'] * (N_STATES - 1) + ['T']     # '------...---T' represents environment (one-dimension world)
    if state == 'terminal':
        interaction = 'Episode %s: total steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction))
        time.sleep(SLEEP_TIME)
    else:
        return
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction))
        time.sleep(FRESH_TIME)

def RL():
    '''
    main part of RL loop
    :return: Q table while all episodes are finished
    '''
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0   # initial poistion
        finished = False
        update_env(state, episode, step_counter)
        while not finished:
            action = choose_action(state, q_table)
            state_, reward = get_env_feedback(state, action)
            former_predict = q_table.loc[state, action]
            if state_ != 'terminal':
                target = reward + GAMMA * q_table.iloc[state_, :].max()
            else:   # next state is terminal
                target = reward
                finished = True
            q_table.loc[state, action] += ALPHA * (target - former_predict) # update Q table
            state = state_

            step_counter += 1
            update_env(state, episode, step_counter)
    return q_table

if __name__ == "__main__":
    q_table = RL()
    print('\r\nQ-table:\n')
    print(q_table)

'''
Episode 1: total steps = 38
Episode 2: total steps = 22
Episode 3: total steps = 9
Episode 4: total steps = 5
Episode 5: total steps = 7
Episode 6: total steps = 5
Episode 7: total steps = 5
Episode 8: total steps = 5
Episode 9: total steps = 5
Episode 10: total steps = 5
Episode 11: total steps = 5
Episode 12: total steps = 7 # due to ramdom choices
Episode 13: total steps = 5

Q-table:

       left     right
0  0.000000  0.004320
1  0.000000  0.025005
2  0.000030  0.111241
3  0.000000  0.368750
4  0.027621  0.745813
5  0.000000  0.000000
'''