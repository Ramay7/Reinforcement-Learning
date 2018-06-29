"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
"""
from maze_env import Maze
from RL_brain import QLearningTable
from RL_brain import SarsaTable
from RL_brain import SarsaLambdaTable

def update_Q_learning():
    for episode in range(100):
        observation = env.reset()   # initial observation
        while True:
            env.render()    # fresh env
            action = RL.choose_action(str(observation))     # RL choose action based on observation
            observation_, reward, done = env.step(action)   # RL take action and get next observation and reward
            RL.learn(str(observation), action, reward, str(observation_))   # RL learn from this transition
            observation = observation_      # swap observation
            if done:    # break while loop when end of this episode
                break
    print('game over')  # end of game
    env.destroy()


def update_sarsa():
    for episode in range(100):
        observation = env.reset()   # initial observation
        action = RL.choose_action(str(observation))     # RL choose current action based on current observation
        while True:
            env.render()    # fresh env
            observation_, reward, done = env.step(action)    # RL obtain next obervation and reward based on current action
            action_ = RL.choose_action(str(observation_))   # RL choose next action based on next observation
            RL.learn(str(observation), action, reward, str(observation_), action_)   # RL learn from this transition
            observation, action = observation_, action_ # replace current observation and action with next observation and action
            if done:
                break
    print('game over')
    env.destroy()


def update_sarsa_lambda():
    for episode in range(100):
        observation = env.reset()       # initial observation
        action = RL.choose_action(str(observation))     # RL choose current action based on current observation
        RL.eligibility_trace *= 0     # initialize eligibility trace
        while True:
            env.render()        # fresh env
            observation_, reward, done = env.step(action)   # RL obtain next observation and reward based on current action
            action_ = RL.choose_action(str(observation_))   # RL choose next action based on next observation
            RL.learn(str(observation), action, reward, str(observation), action_)   # RL learn from this transition
            observation, action = observation_, action_ # replace current observation and action with next observation and action
            if done:
                break
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()

    # Q-Learning Methon
    # RL = QLearningTable(actions=list(range(env.n_actions)))
    # update = update_Q_learning()

    # Sarsa Method
    RL = SarsaTable(actions = list(range(env.n_actions)))
    update = update_sarsa()

    # Sarsa-Lambda Method                     ------- This method does not work very well.
    # RL = SarsaLambdaTable(actions = list(range(env.n_actions)))
    # update = update_sarsa_lambda()

    env.after(100, update)
    env.mainloop()