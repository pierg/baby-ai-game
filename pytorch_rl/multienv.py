import numpy as np

import gym
import gym_minigrid

"""
Plan:
- Multi-env stores single instances of multiple envs
- Returns progress data through info object
- Separate component logs/graphs progress on each env
"""

class MultiEnv(gym.Env):
    """
    Environment which samples from multiple environments, for
    multi-taks learning
    """

    def __init__(self, env_names):
        self.env_names = env_names

        self.env_list = []

        self.action_space = None
        self.observation_space = None

        for env_name in env_names:
            env = gym.make(env_name)

            if self.action_space is None:
                self.action_space = env.action_space

            if self.observation_space is None:
                self.observation_space = env.observation_space

            assert env.action_space == self.action_space
            assert isinstance(env.observation_space, gym.spaces.Dict)

            self.env_list.append(env)

        self.reward_range = (0, 1)

        self.cur_env_idx = 0
        self.cur_reward_sum = 0
        self.cur_num_steps = 0

    def seed(self, seed):
        for env in self.env_list:
            env.seed(seed)

        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        return [seed]

    def reset(self):
        env = self.env_list[self.cur_env_idx]
        return env.reset()

    def step(self, action):
        env = self.env_list[self.cur_env_idx]

        obs, reward, done, info = env.step(action)

        # Normalize the reward into the [0,1] range
        reward_min, reward_max = env.reward_range
        reward = (reward - reward_min) / (reward_max - reward_min)

        # Keep track of the total reward for this episode
        self.cur_reward_sum += reward
        self.cur_num_steps += 1

        # Store the current environment name in the info object
        info['multi_env'] = {}
        info['multi_env']['env_name']= self.env_names[self.cur_env_idx]

        # If the episode is done, sample a new environment
        if done:
            # Add the total reward for the episode to the info object
            info['multi_env']['episode_reward']= self.cur_reward_sum
            info['multi_env']['episode_steps']= self.cur_num_steps
            self.cur_reward_sum = 0
            self.cur_num_steps = 0

            #self.cur_env_idx = self.np_random.randint(0, len(self.env_list))
            self.cur_env_idx = (self.cur_env_idx + 1) % len(self.env_list)
            #print(self.cur_env_idx)


        return obs, reward, done, info

    def render(self, mode='human', close=False):
        env = self.env_list[self.cur_env_idx]
        return env.render(mode, close)

    def close(self):
        for env in self.env_list:
            env.close()

        self.cur_env_idx = 0
        self.env_names = None
        self.env_list = None

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
                    info['episode_steps']
                )

        self.num_updates += 1

        if self.num_updates % 400 == 0:
            self.genMultiGraph()

    def addDataPoint(self, env_name, episode_reward, episode_steps):
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

        if data['num_episodes'] % 100 == 0:
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
        min_num_pts = 0
        for env_name, data in self.env_data.items():
            max_num_pts = max(max_num_pts, len(data['x_values']))
            min_num_pts = min(max_num_pts, len(data['x_values']))

        # If some environment has no data points yet, then wait
        if min_num_pts == 0:
            return

        X = None
        Y = None

        legend = []

        for env_name, data in self.env_data.items():
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

            legend.append(env_name)

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
