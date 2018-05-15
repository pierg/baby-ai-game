import numpy as np

import torch

from configurations import config_grabber as cg

import csv_logger

import os


class Evaluator:
    
    def __init__(self):
        # Getting configuration from file
        self.config = cg.Configuration.grab()

        config_file_path = os.path.abspath(__file__ + "/../../"
                                           + self.config.evaluation_directory_name + "/"
                                           + self.config.config_name
                                           + ".csv")

        dirname = os.path.dirname(config_file_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Setup CSV logging
        csv_logger.create_header(config_file_path,
                                 ['N_updates',
                                  'N_timesteps',
                                  'FPS',
                                  'Reward_mean',
                                  'Reward_median',
                                  'Reward_min',
                                  'Reward_max',
                                  'Entropy',
                                  'Value_loss',
                                  'Action_loss',
                                  'N_episodes',
                                  'N_blocked_actions',
                                  'N_goal_reached'])

        # Evaluation variables
        # self.shortest_path = config.shortest_path
        
        self.episode_rewards = torch.zeros([self.config.num_processes, 1])
        self.final_rewards = torch.zeros([self.config.num_processes, 1])

        self.n_catastrophes = torch.zeros([self.config.num_processes, 1])
        self.n_episodes = torch.zeros([self.config.num_processes, 1])
        self.n_proccess_reached_goal = torch.zeros([self.config.num_processes, 1])



    def update(self, reward, done, info):

        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        self.episode_rewards += reward

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        self.final_rewards *= masks
        self.final_rewards += (1 - masks) * self.episode_rewards
        self.episode_rewards *= masks

        n_catastrophes_mask = torch.FloatTensor([[1.0] if 'violation' in info_ else [0.0] for info_ in info])
        n_episodes_mask = torch.FloatTensor([[1.0] if done_ else [0.0] for done_ in done])
        n_process_reached_goal_mask = torch.FloatTensor([[1.0] if 'goal' in info_ else [0.0] for info_ in info])

        for done__ in done:
            if done__:
                self.n_episodes = self.n_episodes + 1

        self.n_catastrophes += n_catastrophes_mask
        self.n_episodes += n_episodes_mask
        self.n_proccess_reached_goal += n_process_reached_goal_mask


    def save(self, n_updates, t_start, t_end, dist_entropy, value_loss, action_loss):

        total_num_steps = (n_updates + 1) * self.config.num_processes * self.config.num_steps

        csv_logger.write_to_log("{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
            n_updates,
            total_num_steps,
            int(total_num_steps / (t_end - t_start)),
            self.final_rewards.mean(),
            self.final_rewards.median(),
            self.final_rewards.min(),
            self.final_rewards.max(),
            dist_entropy.data[0],
            value_loss.data[0],
            action_loss.data[0],
            self.n_episodes.sum(),
            self.n_catastrophes.sum(),
            self.n_proccess_reached_goal.sum()
        ))