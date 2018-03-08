import math
import pickle

import numpy as np

import gym
import gym_minigrid
from gym_minigrid.minigrid import MiniGridEnv

class DemoEnv(MiniGridEnv):
    def __init__(self, testSet=False):
        self.epsCount = 0
        self.testSet = testSet
        super().__init__(gridSize=19, maxSteps=50)

    def _genGrid(self, width, height):
        if self.epsCount % 200 == 0:
            #print('Loading demonstrations')
            demos = pickle.load(open('demos.p', 'rb'))
            demos = filter(lambda d: d['testSet'] == self.testSet, demos)
            self.demos = list(demos)
            num_demos = len(self.demos)
            #print('num demos: %d' % num_demos)

        self.epsCount += 1

        # Select a random demonstration
        demo = self._randElem(self.demos)

        self.startPos = demo['startPos']
        self.startDir = self._randInt(0, 4)

        self.endGrid = demo['endGrid'].encode()
        self.endPos = demo['endPos']

        self.mission = demo['mission']

        self.testSet = demo['testSet']
        self.demoSteps = demo['numSteps']
        self.maxSteps = 4 * self.demoSteps

        return demo['startGrid'].copy()

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)

        if self.agentPos == self.endPos:
            curGrid = self.grid.encode()
            if np.array_equal(curGrid, self.endGrid):
                # Penalize for taking more steps than in the demonstration
                stepsOver = max(0, self.stepCount - self.demoSteps)
                stepsOverRel = stepsOver / (self.maxSteps - self.demoSteps)
                reward = 1 - stepsOverRel
                done = True

        # Store the current environment name in the info object
        info['multi_env'] = {}
        info['multi_env']['env_name'] = self.mission

        # If the episode is done, add the total reward for the episode to the info object
        if done:
            info['multi_env']['episode_reward'] = reward
            info['multi_env']['episode_steps'] = self.stepCount
            info['multi_env']['test_set'] = self.testSet

        return obs, reward, done, info

class MultiEnvGraphing:
    """
    Code to produce visdom plots of the training progress.

    Start a Visdom server with:
        python -m visdom.server
    The Visdom page will be at:
        http://localhost:8097/
    """

    def __init__(self):
        from visdom import Visdom
        self.vis = Visdom()
        assert self.vis.check_connection()

        # Close existing windows
        self.vis.close()

        # Per-environment data, indexed by environment name
        self.env_data = {}

        self.plot = None

        self.num_updates = 0

    def process(self, infos):
        for info in infos:
            info = info['multi_env']
            env_name = info['env_name']

            if 'episode_reward' in info:
                self.addDataPoint(
                    env_name,
                    info['episode_reward'],
                    info['episode_steps'],
                    info['test_set']
                )

        self.num_updates += 1

        if self.num_updates % 400 == 0:
            self.genMultiGraph()

    def addDataPoint(self, env_name, episode_reward, episode_steps, test_set):
        data = self.env_data.setdefault(
            env_name,
            dict(
                y_values = [],
                x_values = [],
                num_episodes = 0,
                num_steps = 0,
                running_avg = 0,
                plot = None
            )
        )

        data['running_avg'] *= 0.995
        data['running_avg'] += 0.005 * episode_reward

        data['num_episodes'] += 1
        data['num_steps'] += episode_steps

        if test_set or data['num_episodes'] % 100 == 0:
            #data['x_values'].append(data['num_steps'])
            data['x_values'].append(data['num_episodes'])
            data['y_values'].append(data['running_avg'])

            data['plot'] = self.vis.line(
                X = np.array(data['x_values']),
                Y = np.array(data['y_values']),
                opts = dict(
                    #title="Reward per episode",
                    title = env_name,
                    xlabel='Number of episodes',
                    ylabel='Reward per episode',
                    ytickmin=0,
                    ytickmax=1,
                    ytickstep=0.1,
                ),
                win = data['plot']
            )

        return data

    def genMultiGraph(self):
        # Find out the size of the longest dataset
        max_num_pts = 0
        min_num_pts = math.inf
        for env_name, data in self.env_data.items():
            max_num_pts = max(max_num_pts, len(data['x_values']))
            min_num_pts = min(min_num_pts, len(data['x_values']))

        # If some environment has no data points yet, then wait
        if min_num_pts == 0:
            return

        legend = sorted(self.env_data.keys())
        X = None
        Y = None

        for env_name in legend:
            data = self.env_data[env_name]
            x_values = data['x_values']
            y_values = data['y_values']

            pad_len = max_num_pts - len(x_values)
            x_values = np.pad(x_values, (0, pad_len), 'edge')
            y_values = np.pad(y_values, (0, pad_len), 'edge')

            x_values = x_values.reshape((max_num_pts, 1))
            y_values = y_values.reshape((max_num_pts, 1))

            if X is None:
                X = x_values
            else:
                X = np.concatenate((X, x_values), axis=1)

            if Y is None:
                Y = y_values
            else:
                Y = np.concatenate((Y, y_values), axis=1)

        self.plot = self.vis.line(
            #X = np.array(data['x_values']),
            #Y = np.array(data['y_values']),
            X = X,
            Y = Y,
            opts = dict(
                title = 'All Environments',
                xlabel='Number of episodes',
                ylabel='Reward per episode',
                ytickmin=0,
                ytickmax=1,
                ytickstep=0.1,
                legend=legend,
                showlegend=True,
                width=900,
                height=500
            ),
            win = self.plot
        )
