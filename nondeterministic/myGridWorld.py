import numpy as np

BOARD_ROWS = 5
BOARD_COLS = 6
WIN_STATE = (0, 2)
LOSE_STATE = (0, 3)
START = (4, 0)
DETERMINISTIC = False
BLOCK_STATE = (2,2)

# BOARD_ROWS = 3
# BOARD_COLS = 4
# WIN_STATE = (0, 3)
# LOSE_STATE = (1, 3)
# START = (2, 0)
# DETERMINISTIC = False
# BLOCK_STATE = (1,1)

class State:
    def __init__(self, state=START):
        self.state = state
        self.isEnd = False
        self.deterministic = False
        self.obey = 0.8

    def giveReward(self):
        if self.state == WIN_STATE:
            return 1
        elif self.state == LOSE_STATE:
            return -1
        else:
            return 0

    def isEndFunc(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True

    def _chooseActionProb(self,action):
        if action == "up":
            return np.random.choice(["up", "left", "right"], p=[self.obey, (1-self.obey)/2, (1-self.obey)/2])
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=[self.obey, (1-self.obey)/2, (1-self.obey)/2])
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=[self.obey, (1-self.obey)/2, (1-self.obey)/2])
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=[self.obey, (1-self.obey)/2, (1-self.obey)/2])

    
    def nxtPosition(self, action):
        if (self.deterministic):
            if action == "up":
                nxtState = (self.state[0]-1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            self.determine = False

        else:
            action = self._chooseActionProb(action)
            self.deterministic = True
            nxtState = self.nxtPosition(action)

        if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS-1)):
            if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS-1)):
                if nxtState != BLOCK_STATE:
                    return nxtState
        return self.state
    
class Agent:
    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.lr = 0.2
        self.exp_rate = 0.3
        self.decay_gamma = 0.9
        self.Q_values = {}

        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] ={}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0
    
    def chooseAction(self):
        action = ""
        mx_nxt_reward = 0
        #exploration
        if (np.random.uniform(0,1)) <= (self.exp_rate):
            action = np.random.choice(self.actions)
        #exploitation
        else:
            for a in self.actions:
                currentPosition = self.State.state
                nxt_reward = self.Q_values[currentPosition][a]
                if (nxt_reward >= mx_nxt_reward):
                    action = a
                    mx_nxt_reward = nxt_reward
        if(action == ""):
            action = self.chooseAction()
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()

    def play(self, rounds):
        i = 0
        while i < rounds:
            if self.State.isEnd:
                reward = self.State.giveReward()
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward  
                print("--------------Game End Reward----------------------{}".format(i))
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                self.reset()
                i += 1

            else:
                action = self.chooseAction()
                self.states.append([(self.State.state), action])
                self.State = self.takeAction(action)
                self.State.isEndFunc()
                print("nxt state", self.State.state)
                print("---------------------")
    
    def showValues(self, value, policy):
        if(value == 1):
            for i in range(0, BOARD_ROWS):
                print('----------------------------------')
                out = '| '
                for j in range(0, BOARD_COLS):
                    val = self.Q_values[(i,j)]["up"]
                    for a in self.actions:                    
                        if val <= self.Q_values[(i,j)][a]:
                            val = self.Q_values[(i,j)][a]        
                    out += str(val).ljust(6) + ' | '
                print(out)
            print('----------------------------------')
        if(policy == 1):
            for i in range(0, BOARD_ROWS):
                print('----------------------------------')
                out = '| '
                for j in range(0, BOARD_COLS):
                    val = self.Q_values[(i,j)]["up"] 
                    for a in self.actions:     
                        if(i,j) == BLOCK_STATE:
                            action = '0' 
                            break                
                        if val <= self.Q_values[(i,j)][a]:
                            action = a
                            val = self.Q_values[(i,j)][a]        
                    out += action.ljust(6) + ' | '
                print(out)
            print('----------------------------------')


if __name__ == "__main__":
    ag = Agent()
    print("initial Q-values ... \n")
    print(ag.Q_values)

    ag.play(50)
    print("latest Q-values ... \n")
    print(ag.Q_values)
    print(ag.Q_values[(0,0)]['up'])
    ag.showValues(1,0)
    ag.showValues(0,1)