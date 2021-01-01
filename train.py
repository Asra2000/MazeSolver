import numpy as np
import pygame as py
from environment import Environment

# defining the parameters
gamma = 0.9  # the discounting factor
alpha = 0.75  # the learning rate
nEpochs = 10000  # how many times due we want the Q learning take place for one action

# Environment and Q-table initialization
# Q-table holds the value of each state
env = Environment()
rewards = env.rewardBoard

# the rows point to the current state and 
# the col tell about the next state to be taken
QTable = rewards.copy()

# preparing the Q-learning process 1

# steps 1 check where can we cannot start of from i.e wall
possibleStates = []
for i in range(rewards.shape[0]):
    if sum(abs(rewards[i])) != 0:
        # that it is not a wall we can use this position
        possibleStates.append(i)


# prepare the Q-learning process 2
def maximum(qvalues):
    index = 0
    maxQValue = -np.inf
    for i in range(len(qvalues)):
        if qvalues[i] > maxQValue and qvalues[i] != 0:
            maxQValue = qvalues[i]
            index = i

    return index, maxQValue


# starting the Q learning process
for epoch in range(nEpochs):
    print('\rEpoch: ' + str(epoch + 1), end="")

    startingPos = np.random.choice(possibleStates)

    # getting all the playable action
    possibleActions = []

    for i in range(rewards.shape[1]):
        # iterate no. of cell size
        if rewards[startingPos][i] != 0:
            possibleActions.append(i)

    # playing a random action
    action = np.random.choice(possibleActions)

    reward = rewards[startingPos][action]

    # Updating the Q value
    _, maxQvalue = maximum(QTable[action])

    TD = reward + gamma * maxQvalue - QTable[startingPos][action]
    QTable[startingPos][action] = QTable[startingPos][action] + alpha * TD

# displaying the result
currentPos = env.startingPos
while True:
    action, _ = maximum(QTable[currentPos])

    env.movePlayer(action)

    currentPos = action


