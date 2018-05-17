import argparse
import json
from secrets import token_hex

parser = argparse.ArgumentParser(description='Arguments for creating the environments and its configuration')
parser.add_argument('--grid_size', type=int, required=True)
parser.add_argument('--number_of_water_tiles', type=int, required=True)
parser.add_argument('--max_block_size', type=int, required=True)


environment_path = "../gym-minigrid/gym_minigrid/envs/"
configuration_path = "configurations/"
random_token = token_hex(4)


def generate_environment(grid_size, nr_of_water_tiles, max_block_size):
    # Creates a random environment python file, it currently only places water tiles randomly in the environment
    # It uses the random_token as the seed for all its random functions, this to enable reproduction
    # of environments. Using the random_token found in the end of file name.
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
        while 5 > placed_water_tiles:
            # Minus 2 because grid is zero indexed, and the last one is just a wall
            width_pos = random.randint(1, width - 2)
            height_pos = random.randint(1, height - 2)

            if width_pos == 1 and height_pos == 1:
                # Do not place water on agent
                continue
            if width_pos == 1 and height_pos == 2 and isinstance(self.grid.get(2, 1), Water) or \
                width_pos == 2 and height_pos == 1 and isinstance(self.grid.get(1, 2), Water):
                # Do not place two water tiles in front of the agent A W -
                #                                                    W | |
                #                                                    | | |
                continue
            if isinstance(self.grid.get(width_pos, height_pos), Water):
                # Do not place water on water
                continue
            if isinstance(self.grid.get(width - 2, height - 2), Goal):
                # Do not place water on Goal
                continue
            if width_pos == width - 2 and height_pos == height - 3 and isinstance(self.grid.get(width - 3, height - 2)) or \
                width_pos == width - 3 and height_pos == height - 2 and isinstance(self.grid.get(width - 2, height - 3), Water):
                # Do not place water preventing the agent from reaching the goal - | |
                #                                                                - A W
                #                                                                - W G
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
    # Adds the import statement to __init__.py in the envs folder in gym_minigrid,
    # otherwise the environment is unavailable to use.
    with open(environment_path + "__init__.py", 'a') as init_file:
        init_file.write("\n")
        init_file.write("from gym_minigrid.envs.randomenv{0}{1} import *".format(nr_of_water_tiles, random_token))
        init_file.close()

    # Creates a json config file for the random environment
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

    # Updates the main.json file and set the env name to the randomly generated one
    with open(configuration_path + "main.json", 'r+') as main_config:
        data = json.load(main_config)
        data["env_name"] = "MiniGrid-RandomEnv-{0}x{0}-v0".format(grid_size)
        main_config.seek(0)
        json.dump(data, main_config, sort_keys=True, indent=4)
        main_config.close()

    return "RandomEnv-{0}x{0}-Water-{1}-{2}".format(grid_size, nr_of_water_tiles, random_token)


def main():
    args = parser.parse_args()
    print(args)
    file_name = generate_environment(args.grid_size, args.number_of_water_tiles, args.max_block_size)
    print(file_name)


if __name__ == '__main__':
    main()
