import random
import numpy as np
from typing import Tuple
from copy import deepcopy
from config import *
if use_keras:
    from keras.utils.np_utils import to_categorical
from numpy.random import choice
class State:
    def __init__(self,board,pos,prev_position):
        self.board = board
        self.pos = pos
        self.prev_position = prev_position
        
        self.feed=np.concatenate((self.board,pos,prev_position))
        
class Game:
        def __init__(self):
            self.origin = np.zeros(2)
            self.robot = np.zeros(2)
            self.board = None
            self.og_board = None
            self.prev_position = np.zeros(2)
            self.prev_action = np.zeros(2)
        def state(self) -> State:
            return State(np.copy(self.board),self.robot,self.prev_action)
        def __getitem__(self, item):
            return np.argmax(self.board[int(item[1])][int(item[0])],axis=0)
        def __setitem__(self, item, value):
            self.board[int(item[1])][int(item[0])] = to_categorical(value,num_classes=4)
        def __repr__(self):
            s=""
            for y in range(len(self.board)):
                for x in range(len(self.board)):
                    v = self[x,y]
                    if self.robot[0]==x and self.robot[1]==y:
                        s+="|R|"
                    else:
                        s+= ["-","Z","B","E"][v]
                    s+="\t"
                s+="\n"
            return s
        def generate(self,dimensions):
            self.board = np.zeros((dimensions,dimensions,4))
            for y in range(dimensions):
                for x in range(dimensions):
                    c = choice([2,1,0],1,p=[.1,.2,.7])[0]
                    cat = to_categorical(c,num_classes=4)
                    self.board[y][x]=cat
            self[dimensions-1,dimensions-1] =3
            self.og_board =np.copy(self.board)
            self[self.origin]=0
        def move(self,dir)->Tuple[State,int,bool]:
            og_pos = self.robot
            reward,end = self.stateless_move(dir)
            return State(np.copy(self.board),self.robot,og_pos),reward,end
        def stateless_move(self,dir):
            self.prev_position = np.copy(self.robot)
            self.prev_action=np.copy(dir)
            self.robot = np.array([self.robot[0]+dir[0],self.robot[1]+dir[1]])
            if self.robot[0]<0 or self.robot[0]>=len(self.board) or self.robot[1]<0 or self.robot[1]>=len(self.board):
                self.reset()
                return -100,True
            if self[self.robot]==2:
                self.reset()
                return -100,True
            if self[self.robot]==1:
                self[self.robot]=0
                return 3, False
            if self[self.robot]==0:
                return -1,False
            if self[self.robot]==3:
                self.reset()
                return 100,False
            raise Exception("bruh")

        def reset(self):
            self.robot = np.zeros(2)
            self.board = np.copy(self.og_board)
            self.prev_position=np.zeros(2)
            self.prev_action=np.zeros(2)


def hash_np(n):
    return str(n)

def normalize(arr):
    m = min(arr)
    narr =[k-m for k in arr]
    s = sum(narr)
    if s==0:
        return [1/len(arr) for k in arr]
    return [k/s for k in narr]

class Table:
    DIRS = ((0, 1), (1, 0), (-1, 0), (0, -1))

    def __init__(self,lr=1):
        self.tables = {}
        self.lr = lr

    def decide(self,state,epsilon):
        if hash_np(state) not in self.tables:
            self.tables[hash_np(state)] = {k:0 for k in self.DIRS}
            return random.choice(self.DIRS)
        if random.uniform(0,1)<epsilon:
            return random.choice(self.DIRS)
        return max(self.tables[hash_np(state)].items(),key=lambda x:x[1])[0]
    def init_state(self,state):
        if hash_np(state) not in self.tables:
            self.tables[hash_np(state)] = {k:0 for k in self.DIRS}
    def expected_reward(self,state):
        self.init_state(state)
        return max(self.tables[hash_np(state)].values())
    def update(self,previous_state,new_state,action,reward,discount):
        self.tables[hash_np(previous_state)][action]+=self.lr*(reward+discount*self.expected_reward(new_state)-self.tables[hash_np(previous_state)][action])

def run():
    table = Table()
    game = Game()
    game.generate(15)
    print(game)
    step_counter = 0
    dop = False
    epsilon = .3
    while True:
        og_state = game.robot
        decision= table.decide(game.robot,epsilon)
        reward, reset=  game.stateless_move(decision)
        step_counter+=reward
        if reset:
            step_counter=0
        if reward==100:
            print("CONGRATULATIONS steps:",step_counter)
            step_counter=0
        table.update(og_state,game.robot,decision,reward,.8)
        if dop:
            print(game)

# run()