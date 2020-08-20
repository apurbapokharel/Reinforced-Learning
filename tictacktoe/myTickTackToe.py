import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3

class Player:
    def __init__(self, name, exp_rate, lr_rate):
        self.name = name
        self.exp_rate = exp_rate
        self.lr_rate = lr_rate
        self.state = []
        self.stateValues = {}
        self.decay = 0.9
        self.winRate = 0

    def setSymbol(self, symbol):
        self.symbol = symbol

    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash
        
    def addState(self, boardHash):
        self.state.append(boardHash)
        
    def savePolicy(self):
        fw = open('policy_' + str(self.symbol), 'wb')
        pickle.dump(self.stateValues, fw)
        fw.close()

    def loadPolicy(self):
        fr = open('policy_' + str(self.symbol),'rb')
        self.stateValues = pickle.load(fr)
        fr.close()

    def reset(self):
        self.state = []
    
    def feed(self, positions, board):
        self.positions = positions
        self.board = board

    def feedReward(self, reward):
        for st in reversed(self.state):
            if self.stateValues.get(st) is None:
                self.stateValues[st] = 0 
            self.stateValues[st] += self.lr_rate * (self.decay * reward - self.stateValues[st])
            reward = self.stateValues[st]
        self.reset()

    def chooseAction(self):
        if np.random.uniform(0,1) <= self.exp_rate:
            index = np.random.choice(len(self.positions))
            action = self.positions[index]
        else:
            max = -999
            for position in self.positions:
                copyBoard = self.board.copy()
                copyBoard[position] = self.symbol
                boardHash = self.getHash(copyBoard)
                stValue = 0 if self.stateValues.get(boardHash) is None else self.stateValues.get(boardHash) 
                if stValue >= max:
                    action = position
                    max = stValue
        return action

class  HumanPlayer:
    def __init__(self):
        self.name = "Human"
        self.winRate = 0
    
    def feed(self, positions, board):
        self.positions = positions
    
    def chooseAction(self):
        while True:
            position = int(input("Enter position:"))
            position -= 1
            i = position // int(BOARD_COLS)
            j = position % BOARD_COLS
            action = (i, j)
            if action in self.positions:
                return action

    def feedReward(self, reward):
        print("human recives ", reward)

    def setSymbol(self, symbol):
        self.symbol = symbol

    def addState(self, boardHash):
        pass

class State:
    def __init__(self, p1, p2, competeStatus):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.player1 = p1
        self.player2 = p2
        self.isEnd = False
        self.currentPlayer = self.player1 #player1 plays first
        self.player1.setSymbol(1)
        self.player2.setSymbol(-1)
        self.isCompeteBot = competeStatus

    def getHash(self):
        boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def getPositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i,j] == 0:
                    positions.append((i,j))
        return positions

    def winner(self):
        for i in range(BOARD_ROWS):
            #row
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1
            #cols
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1
        #diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        if diag_sum1 == 3 or diag_sum2 == 3:
            self.isEnd = True
            return 1
        if diag_sum1 == -3 or diag_sum2 == -3:
            self.isEnd = True
            return -1
        if len(self.getPositions()) == 0:
            self.isEnd = True
            return 0
        else:
            self.isEnd = False
            return None

    def takeAction(self, action):
       self.board[action] = self.currentPlayer.symbol 
           
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.isEnd = False
        self.currentPlayer = self.player1

    def checkWin(self):
        winStatus = self.winner()
        if winStatus == self.player1.symbol:
            if not self.isCompeteBot:
                self.player1.feedReward(1) 
                self.player2.feedReward(0)
            self.player1.winRate += 1
            self.isEnd = True

        if winStatus == self.player2.symbol:
            if not self.isCompeteBot:
                self.player1.feedReward(0) 
                self.player2.feedReward(1)
            self.player2.winRate += 1
            self.isEnd = True

        if winStatus == 0: #tie 
            if not self.isCompeteBot:
                self.player1.feedReward(0.1) 
                self.player2.feedReward(0.5)
            self.isEnd = True

        if winStatus == None: #ongoing so, change player
            if self.currentPlayer.symbol == 1:
                self.currentPlayer = self.player2
            else:
                self.currentPlayer = self.player1

    def feedValues(self, postions, board):
        self.currentPlayer.feed(postions, board)

    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')

    def play(self, show=False):
        self.reset()
        while not self.isEnd:
            if show:
                self.showBoard()
            #p1 
            positions = self.getPositions()
            self.feedValues(positions, self.board)
            action = self.currentPlayer.chooseAction()
            self.takeAction(action)
            if not self.isCompeteBot:
                boardHash = self.getHash()
                self.currentPlayer.addState(boardHash)
            self.checkWin()
            if self.isEnd:
                break
            if show:
                self.showBoard()
            #p2 
            positions = self.getPositions()
            self.feedValues(positions, self.board)
            action = self.currentPlayer.chooseAction()
            self.takeAction(action)
            if not self.isCompeteBot:
                boardHash = self.getHash()
                self.currentPlayer.addState(boardHash)
            self.checkWin()
            if show:
                self.showBoard()
          
def train(rounds=15000):
    print("Training")
    player1 = Player("p1", 0.3, 0.2)
    player2 = Player("p2", 0.3, 0.2)
    st = State(player1, player2, False)
    for i in range(rounds):
        if i % 1000 == 0:
            print("Rounds", i)
        st.play()
    tieRate = (rounds-(player1.winRate)-player2.winRate)
    print("player 1 won {} out of {} games i.e {} winRate".format(player1.winRate, rounds, player1.winRate/rounds))
    print("player 2 won {} out of {} games i.e {} winRate".format(player2.winRate, rounds, player2.winRate/rounds))
    print("games tied {} times i.e {} tieRate".format(tieRate, tieRate/rounds))
    player1.savePolicy()
    player2.savePolicy()

def compete(rounds=100):
    print("Competing")
    player1 = Player("p1", exp_rate=0, lr_rate=0)
    player2 = Player("p2", exp_rate=0, lr_rate=0)
    st = State(player1, player2, True)
    player1.loadPolicy()
    player2.loadPolicy()
    for i in range(rounds):
        st.play()
    tieRate = (rounds-(player1.winRate)-player2.winRate)
    print("player 1 won {} out of {} games i.e {} winRate".format(player1.winRate, rounds, player1.winRate/rounds))
    print("player 2 won {} out of {} games i.e {} winRate".format(player2.winRate, rounds, player2.winRate/rounds))
    print("games tied {} times i.e {} tieRate".format(tieRate, tieRate/rounds))
    
def play():
    exit = 1
    while exit:
        player1 = Player("p1", exp_rate=0, lr_rate=0.35)
        player2 = HumanPlayer()
        st = State(player1, player2, False)
        # st = State(player2, player1, False)
        player1.loadPolicy()
        st.play(True)
        player1.savePolicy()
        exit = int(input("Enter 1 to keep playing or 0 to quit:"))

if __name__ == "__main__":
    train()
    compete()
    play()
