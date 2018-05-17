import argparse
import json

parser = argparse.ArgumentParser(description='Arguments for creating the environments and its configuration')
parser.add_argument('--grid_size', type=int)
parser.add_argument('--number_of_water_tiles', type=int)
parser.add_argument('--max_block_size', type=int)

from secrets import token_hex  # NOQA

environment_path = "../gym-minigrid/gym_minigrid/envs/"
configuration_path = "configurations/"
random_token = token_hex(4)


def generate_environment(grid_size, nr_of_water_tiles, max_block_size):
    with open(environment_path + "randomenv{0}{1}.py".format(nr_of_water_tiles, random_token), 'w') as env:
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
        random.seed("{2}")

        # Place water
        placed_water_tiles = 0
        while {1} > placed_water_tiles:
            # Minus 2 because grid is zero indexed, and the last one is just a wall
            width_pos = random.randint(1, width - 2)
            height_pos = random.randint(1, height - 2)
            
            if width_pos == 1 and height_pos == 1:
                # Do not place water on agent
                continue
            if isinstance(self.grid.get(width_pos, height_pos), Water):
                # Do not place water on water
                continue                
        
            self.grid.set(width_pos, height_pos, Water())
            placed_water_tiles += 1
        self.mission = ""

class RandomEnv{0}x{0}(RandomEnv):
    def __init__(self):
        super().__init__(size={0})

register(
    id='MiniGrid-RandomEnv-{0}x{0}-v0',
    entry_point='gym_minigrid.envs:RandomEnv{0}x{0}'
)
""".format(grid_size, nr_of_water_tiles, random_token))
        env.close()
    with open(environment_path + "__init__.py", 'a') as init_file:
        init_file.write("from gym_minigrid.envs.randomenv{0}{1} import *".format(nr_of_water_tiles, random_token))
        init_file.close()

    with open(configuration_path + "randomEnv-{0}-{1}.json".format(nr_of_water_tiles, random_token), 'w') as config:
        config.write(json.dumps({
            "config_name": "firstEvalWaterRandomEnv",
            "algorithm": "a2c",
            "monitors": {
                "absence": {
                    "monitored": {
                        "water": {
                            "active": True,
                            "name": "water",
                            "reward": {
                                "near": "0",
                                "immediate": "-15",
                                "violated": "-55"
                            }
                        }
                    },
                },
            },
            "env_name": "MiniGrid-RandomEnv-{0}x{0}-v0".format(grid_size),
            "num_processes": "48",
            "num_steps": "5",
            "log_interval": "10",
            "on_violation_reset": False,
            "rendering": False,
            "evaluation_directory_name": "evaluations",
            "visdom": False,
            "debug_mode": False,
            "reward": {
                "goal": "1000",
                "step": "-1"
            }
        }))
        config.close()
    return "RandomEnv-{0}x{0}-Water-{1}-{2}".format(grid_size, nr_of_water_tiles, random_token)


def main():
    args = parser.parse_args()
    print(args)
    file_name = generate_environment(args.grid_size, args.number_of_water_tiles, args.max_block_size)
    print(file_name)


if __name__ == '__main__':
    main()
