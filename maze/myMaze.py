import pygame
import numpy as np
import random
from time import sleep

block = [(3, 0), (1, 1), (6, 1), (2, 2), (5, 2), (2, 3), (6, 3), (0, 4), (1, 4), (6, 4), (6, 5), (3, 5), (1, 6), (3, 6), (4, 6), (6, 6)]
end = (6, 7)
start = (0, 0)
class State:
    def __init__(self, currentPos = start):
        self.currentPos = currentPos
        self.n = 8
        self.scrx = 780
        self.scry = 780
        self.background = (51,51,51) 
        self.colors = [(51,51,51) for i in range( self.n**2)] 
        self.terminals = []
        self.reward = {}
        
        for a in range(self.n):
            for b in range(self.n):
                self.reward[(a, b)] = 0
                if (a, b) == end:
                    self.reward[(a, b)] = 1
                    self.colors[self.n * a + b] = (0,255,0)              
        for a in block:
            self.reward[a[0], a[1]] = -1
            self.colors[self.n*a[0] + a[1]] = (255,0,0)

        self.terminals.append(start)
        self.terminals.append(end)
        for a in block:
            self.terminals.append(a)

        pygame.init()
        infoObject = pygame.display.Info()
        if self.scry % self.n != 0:
            a = self.scry % self.n
            self.scry -= a
            self.scrx = self.scry
        self.screen = pygame.display.set_mode(( self.scrx,  self.scry))

    def reset(self):
        self.currentPos = start    

    def nxtPosition(self, action):
            if action == "up":
                nxtState = (self.currentPos[0]-1, self.currentPos[1])
            elif action == "down":
                nxtState = (self.currentPos[0] + 1, self.currentPos[1])
            elif action == "left":
                nxtState = (self.currentPos[0], self.currentPos[1] - 1)
            else:
                nxtState = (self.currentPos[0], self.currentPos[1] + 1)

            return nxtState

    def layout(self):
        c = 0
        a = self.scrx / self.n
        for i in range(0,self.scrx,int(a)):
            for j in range(0,self.scry,int(a)):
                pygame.draw.rect(self.screen,(255,255,255),(j,i,j+int(a),i+int(a)),0)
                pygame.draw.rect(self.screen,self.colors[c],(j+10,i+10,j+int(a)-10,i+int(a) -10),0)
                c+=1
                pygame.draw.circle(self.screen,(25,129,230),(self.currentPos[1]*int(a) + int(a/2) ,self.currentPos[0]*int(a) + int(a/2)),abs(int(a/3) ),0)

class Agent:
    def __init__(self):
        self.state = State()
        self.epsilon = 0.25
        self.lrRate = 0.1
        self.gamma = 0.9
        self.actions = ["up", "down", "left", "right"]
        self.qValue = {}
        for a in range(self.state.n):
            for b in range(self.state.n):
                self.qValue[(a, b)] = {}
                for i in self.actions:
                    self.qValue[(a, b)][i] = 0

    def chooseAction(self):
        possible_actions = []
        currentPosition = self.state.currentPos
        n = self.state.n
        if np.random.uniform() <= self.epsilon:
            if currentPosition[0] != 0:
                possible_actions.append("up")
            if currentPosition[0] != n-1:
                possible_actions.append("down")
            if currentPosition[1] != 0:
                possible_actions.append("left")
            if currentPosition[1] != n-1:
                possible_actions.append("right")
            action = np.random.choice(possible_actions)
        else:
            m = -999
            if currentPosition[0] != 0: #up
                possible_actions.append(self.qValue[currentPosition]["up"])
            else:
                possible_actions.append(m)
            if currentPosition[0] != n-1: #down
                possible_actions.append(self.qValue[currentPosition]["down"])
            else:
                possible_actions.append(m)
            if currentPosition[1] != 0: #left
                possible_actions.append(self.qValue[currentPosition]["left"])
            else:
                possible_actions.append(m)
            if currentPosition[1] != n-1: #right
                possible_actions.append(self.qValue[currentPosition]["right"])
            else:
                possible_actions.append(m)
            index = random.choice([i for i,a in enumerate(possible_actions) if a == max(possible_actions)]) #randomly selecting one of all possible actions with maximun value
            if index == 0:
                action = "up"
            elif index == 1:
                action = "down"
            elif index == 2:
                action = "left"
            else:
                action = "right"
        return action

    def takeAction(self, action):
        position = self.state.nxtPosition(action)
        return State(currentPos=position)

    def getMaxQ(self, position):
        currentState = position
        a = []
        for i in self.actions:
            b = self.qValue[currentState][i]
            a.append(b)
        return(np.max(a))

    def play(self):
        currentState = self.state.currentPos
        action = self.chooseAction()
        self.state = self.takeAction(action)
        newState = self.state.currentPos

        if newState not in self.state.terminals:
            newStateMax = self.getMaxQ(newState)
            self.qValue[currentState][action] += self.lrRate * (self.state.reward[newState] + self.gamma * (newStateMax) - self.qValue[currentState][action])
        else:
            self.qValue[currentState][action] += self.lrRate * (self.state.reward[newState]  - self.qValue[currentState][action])
            self.state.reset()
            if self.epsilon > 0.05:
                self.epsilon -= 3e-4 #reducing as time increases to satisfy Exploration & Exploitation Tradeoff
            
if __name__ == "__main__":
    ag = Agent()
    run = True
    while run:
        ag.state.screen.fill(ag.state.background)
        ag.state.layout()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("quiting.................................................")
                run = False
        pygame.display.flip()
        ag.play()
    pygame.quit()