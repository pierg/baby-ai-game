import pickle

import numpy as np

import gym
import gym_minigrid
from gym_minigrid.minigrid import MiniGridEnv

class DemoEnv(MiniGridEnv):
    def __init__(self):
        self.epsCount = 0
        super().__init__(gridSize=19, maxSteps=50)

    def _genGrid(self, width, height):
        if self.epsCount % 200 == 0:
            #print('Loading demonstrations')
            self.demos = pickle.load(open('demos.p', 'rb'))
            num_demos = len(self.demos)
            print('num demos: %d' % num_demos)

        self.epsCount += 1

        demo = self._randElem(self.demos)

        self.startPos = demo['startPos']
        self.startDir = self._randInt(0, 4)

        self.endGrid = demo['endGrid'].encode()
        self.endPos = demo['endPos']

        self.mission = demo['mission']

        self.maxSteps = 4 * demo['numSteps']

        # TODO: adjust maxSteps

        return demo['startGrid'].copy()

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.agentPos == self.endPos:
            curGrid = self.grid.encode()
            if np.array_equal(curGrid, self.endGrid):
                reward = 1
                done = True

        return obs, reward, done, info
