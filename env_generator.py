import argparse
parser = argparse.ArgumentParser(description='Arguments for creating the environments and its configuration')
parser.add_argument('--grid_size', type=int)
parser.add_argument('--number_of_water_tiles', type=int)
parser.add_argument('--max_block_size', type=int)

from secrets import token_hex  # NOQA

environment_path = "../gym-minigrid/gym_minigrid/envs/"
configuration_path = "configurations/"
random_token = token_hex(8)


def generate_environment(grid_size, nr_of_water_tiles):
    with open(environment_path + "randomenv.py-{0}-{1}".format(nr_of_water_tiles, random_token.decode('utf-8')), 'w') as env:
        env.write("""
from gym_minigrid.extendedminigrid import *
from gym_minigrid.register import register
from random import randint
from random import seed

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

        random_nr = seed()

        # Place water
        placed_water_tiles = 0
        while {1} > index:
            # Minus 2 because grid is zero indexed, and the last one is just a wall
            width_pos = random_nr.randint(1, width - 2)
            height_pos = random_nr.randint(1, height - 2)
            
            if width_pos = 1 and height_pos = 1:
                # Do not place water on agent
                continue
            if isinstance(self.grid.get(width_pos, height_pos), Water()):
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
""".format(grid_size, nr_of_water_tiles))
        env.close()
    with open(configuration_path + "randomEnv-{0}-{1}.json".format(nr_of_water_tiles, random_token.decode('utf-8')), 'w') as config:
        config.write(""" 
{
  "config_name": "firstEvalWaterRandomEnv",
  "algorithm": "a2c",
  "monitors": {
    "absence": {
      "monitored": {
        "water": {
          "active": true,
          "name": "water",
          "reward": {
            "near": 0,
            "immediate": -15,
            "violated": -55
          }
        }
      }
    },
  },
  "env_name": "MiniGrid-RandomEnv-{0}x{0}-v0",
  "num_processes": 48,
  "num_steps": 5,
  "log_interval": 10,
  "on_violation_reset": false,
  "rendering": false,
  "evaluation_directory_name": "evaluations",
  "visdom": false,
  "debug_mode": false,
  "reward": {
    "goal": 1000,
    "step": -1
  }
}""".format(grid_size))
        config.close()
    return "RandomEnv-{0}x{0}-Water-{1}-{2}".format(grid_size, nr_of_water_tiles, random_token.decode('utf-8'))


def main():
    args = parser.parse_args()
    generate_environment(args.grid_size, args.number_of_water_tiles, args.max_block_size)
