import os
import gym
from gym import spaces

import sys

import teacher
import gym_minigrid
from baselines import bench


def make_env(env_id, seed, rank):
    def _thunk():
        
        if env_id=='MultiRoom-Teacher':
            print('creation...')
            env=gym.make('MiniGrid-MultiRoom-N6-v0')
            print('adding Teacher...')
            env=teacher.Teacher(env)
            print('done!')
        else:
            env = gym.make(env_id)

        env.seed(seed + rank)
        #if log_dir is not None:
        #    env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        # Maxime: until RL code supports dict observations, squash observations into a flat vector
        #if isinstance(env.observation_space, spaces.Dict):
        #    print('dic state not supported. we use a Flat wrapper')
        #    env = FlatObsWrapper(env)
        env=WrapPyTorch(env)
        return env

    return _thunk

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

    def observation(self, observation):
        #print('observation', observation)
        return {'image':observation['image'].transpose(2, 0, 1),'mission':observation['mission'],'bestActions':observation['bestActions']}
    
    #def _observation(self, observation):
        #return observation['image'].transpose(2, 0, 1)
    
    
    
