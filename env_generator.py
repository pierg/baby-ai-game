import argparse
import json
from random import randint
from configurations.config_grabber import Configuration

parser = argparse.ArgumentParser(description='Arguments for creating the environments and its configuration')
parser.add_argument('--environment_file', type=str, required=False, help="A json file containing the keys: "
                                                                      "step, goal, near, immediate, violated. "
                                                                      "The values should be the wanted rewards "
                                                                      "of the actions")
parser.add_argument('--rewards_file', type=str, required=False, help="A json file containing the keys: "
                                                                      "step, goal, near, immediate, violated. "
                                                                      "The values should be the wanted rewards "
                                                                      "of the actions")

environment_path = "../gym-minigrid/gym_minigrid/envs/"
configuration_path = "configurations/"
random_token = randint(0,9999)

""" This script creates a random environment in the gym_minigrid/envs folder. It uses a token_hex(4) 
        as the ID and the random seed for placing tiles in the grid.
    This to ensure that certain environments can be reproduced 
        in case the agent behaves strange in certain environments, in order to investigate why.        
"""

def generate_environment(environment="default", rewards="default"):
    elements = Configuration.grab("environments/"+environment)
    grid_size = elements.grid_size
    n_water = elements.n_water
    n_deadend = elements.n_deadend
    light_switch = elements.light_switch
    with open(environment_path + "randoms/" + "randomenv{0}.py".format(random_token), 'w') as env:
        env.write("""
from gym_minigrid.extendedminigrid import *
from gym_minigrid.register import register
import random

class RandomEnv(ExMiniGridEnv):

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (1, 1)
        self.start_dir = 0

        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        # Set the random seed to the random token, so we can reproduce the environment
        random.seed("{4}")

        # Place water
        placed_water_tiles = 0
        while {1} > placed_water_tiles:
            # Minus 2 because grid is zero indexed, and the last one is just a wall
            width_pos = random.randint(1, width - 2)
            height_pos = random.randint(1, height - 2)

            if width_pos == 1 and height_pos == 1:
                # Do not place water on agent
                continue
            if width_pos == 1 and height_pos == 2 and isinstance(self.grid.get(2, 1), Water) or width_pos == 2 and height_pos == 1 and isinstance(self.grid.get(1, 2), Water):
                # Do not place two water tiles in front of the agent A W -
                #                                                    W | |
                #                                                    | | |
                continue
            if isinstance(self.grid.get(width_pos, height_pos), Water):
                # Do not place water on water
                continue
            if isinstance(self.grid.get(width_pos, height_pos), Goal):
                # Do not place water on Goal
                continue
            if width_pos == width - 2 and height_pos == height - 3 and isinstance(self.grid.get(width - 3, height - 2), Water) or width_pos == width - 3 and height_pos == height - 2 and isinstance(self.grid.get(width - 2, height - 3), Water):
                # Do not place water preventing the agent from reaching the goal - | |
                #                                                                - A W
                #                                                                - W G
                continue
            self.grid.set(width_pos, height_pos, Water())
            placed_water_tiles += 1
        self.mission = ""

class RandomEnv{0}x{0}_{4}(RandomEnv):
    def __init__(self):
        super().__init__(size={0})

register(
    id='MiniGrid-RandomEnv-{0}x{0}-{4}-v0',
    entry_point='gym_minigrid.envs:RandomEnv{0}x{0}_{4}'
)
""".format(grid_size, n_water, n_deadend, light_switch, random_token))
        env.close()
    # Adds the import statement to __init__.py in the envs folder in gym_minigrid,
    # otherwise the environment is unavailable to use.
    with open(environment_path + "__init__.py", 'a') as init_file:
        init_file.write("\n")
        init_file.write("from gym_minigrid.envs.randoms.randomenv{0} import *".format(random_token))
        init_file.close()

    # Creates a json config file for the random environment
    with open(configuration_path + "randoms/" + "randomEnv-{0}x{0}-{1}-v0.json".format(grid_size, random_token), 'w') as config:
        rewards = Configuration.grab("rewards/"+rewards)
        config.write(json.dumps({
            "config_name": "randomEnv-{0}x{0}-{1}-v0".format(grid_size, random_token),
            "algorithm": "a2c",
            "monitors": {
                "properties": {
                    "avoid": {
                        "water": {
                            "type": "avoid",
                            "mode": elements.monitors.properties.avoid.water.mode,
                            "action_planner": elements.monitors.properties.avoid.water.action_planner,
                            "active": True,
                            "obj_to_avoid": elements.monitors.properties.avoid.water.obj_to_avoid,
                            "act_to_avoid": elements.monitors.properties.avoid.water.act_to_avoid,
                            "rewards": {
                                "near": float("{0:.2f}".format(rewards.avoid['near'] if 'near' in rewards.avoid else 0)),
                                "immediate": float(
                                    "{0:.2f}".format(rewards.avoid['immediate'] if 'immediate' in rewards.avoid else 0)),
                                "violated": float(
                                    "{0:.2f}".format(rewards.avoid['violated'] if 'violated' in rewards.avoid else -1))
                            }
                        }
                    },
                }
            },
            "env_name": "MiniGrid-RandomEnv-{0}x{0}-{1}-v0".format(grid_size, random_token),
            "num_processes": 48,
            "num_steps": 5,
            "log_interval": 10,
            "on_violation_reset": False,
            "rendering": False,
            "evaluation_directory_name": "evaluations",
            "visdom": False,
            "debug_mode": False,
            "rewards": {
                "standard":{
                    "goal": float("{0:.2f}".format(rewards.standard['goal'] if 'goal' in rewards.standard else 1)),
                    "step": float("{0:.2f}".format(rewards.standard['step'] if 'step' in rewards.standard else 0)),
                    'death': float("{0:.2f}".format(rewards.standard['death'] if 'death' in rewards.standard else -1))
                },
                "cleaningenv":{
                    "clean":float("{0:.2f}".format(rewards.cleaningenv['goal'] if 'clean' in rewards.cleaningenv else 0.5))
                }
            }
        }, indent=2))
        config.close()

    return "randomEnv-{0}x{0}-{1}-v0.json".format(grid_size, random_token)


def main():
    args = parser.parse_args()
    environment = "default"
    rewards = "default"
    if args.rewards_file is not None:
       rewards = args.rewards_file
    if args.environment_file is not None:
        environment = args.environment_file
    file_name = generate_environment(environment, rewards)
    print(file_name)


if __name__ == '__main__':
    main()