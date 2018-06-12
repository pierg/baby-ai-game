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

        # Place dead ends
        placed_dead_ends = 0
        tmp = random.randint(0,3)
        while {2} > placed_dead_ends:
            if self.grid_size < 10:
                # Limit to one dead end if the grid is too small
                if random.randint(1,2) == 1:
                    self.grid.vert_wall(width//2-1,height//2,height//2)
                else:
                    self.grid.horz_wall(width//2,height//2-1,width//2)
                placed_dead_ends = {2}
            else:
                if tmp == 0:
                    self.grid.vert_wall(2,height-6,3)
                    self.grid.horz_wall(1,height-3,1)
                elif tmp == 1:
                    self.grid.horz_wall(1,height-3,3)
                elif tmp == 2:
                    self.grid.vert_wall(width-3,3,3)
                    self.grid.horz_wall(width-2,6,2)
                elif tmp == 3:
                    self.grid.vert_wall(6,0,2)
                    self.grid.horz_wall(3,2,3)
                tmp = (tmp+1)%4
                placed_dead_ends += 1

        # Place water
        placed_water_tiles = 0
        anti_loop = 0
        while {1} > placed_water_tiles:
        
            # Added to avoid a number of water tiles that is impossible (infinite loop)
            anti_loop +=1
            if anti_loop > 1000:
                placed_water_tiles = {1}
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
            if isinstance(self.grid.get(width_pos-1, height_pos), Wall):
                # Do not place water preventing the agent from going into a tunnel
                continue
            if isinstance(self.grid.get(width_pos+1, height_pos), Wall):
                # Do not place water preventing the agent from going into a tunnel
                continue
            if isinstance(self.grid.get(width_pos, height_pos-1), Wall):
                # Do not place water preventing the agent from going into a tunnel
                continue
            if isinstance(self.grid.get(width_pos, height_pos+1), Wall):
                # Do not place water preventing the agent from going into a tunnel
                continue
            if isinstance(self.grid.get(width_pos-1, height_pos-1), Wall):
                # Do not place water preventing the agent from going into a tunnel
                continue
            if isinstance(self.grid.get(width_pos-1, height_pos+1), Wall):
                # Do not place water preventing the agent from going into a tunnel
                continue
            if isinstance(self.grid.get(width_pos+1, height_pos-1), Wall):
                # Do not place water preventing the agent from going into a tunnel
                continue
            if isinstance(self.grid.get(width_pos+1, height_pos+1), Wall):
                # Do not place water preventing the agent from going into a tunnel
                continue
            if isinstance(self.grid.get(width_pos, height_pos), Wall):
                # Do not place water preventing the agent from going into a tunnel
                continue
            self.grid.set(width_pos, height_pos, Water())
            if self.grid_size < 10 and {2} > 0:
                placed_water_tiles = {1}
            else:
                placed_water_tiles += 1
                
        self.mission = ""

    def step(self,action):
        # Reset if agent step on water without knowing it
        if action == self.actions.forward and self.worldobj_in_agent(1,0) == "water" :
            return self.gen_obs(), 0, True, "died"
        else:
            return super().step(action)

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
        list_of_json_properties = {}
        list_of_json_patterns = {}
        properties_map = {}
        patterns_map = {}
        if hasattr(elements,"monitors"):
            if hasattr(elements.monitors,"properties"):
                for type in elements.monitors.properties:
                    for monitor in type:
                        type_of_monitor = monitor.type
                        near = 0
                        immediate = 0
                        violated = -1
                        for current_monitor in rewards:
                            if hasattr(current_monitor, "name"):
                                if current_monitor.name == type_of_monitor:
                                    near = current_monitor.near
                                    immediate = current_monitor.immediate
                                    violated = current_monitor.violated
                        list_of_json_properties[monitor.name] = {
                                "{0}".format(monitor.name): {
                                    "type": "{0}".format(monitor.type),
                                    "mode": "{0}".format(monitor.mode),
                                    "action_planner": "{0}".format(monitor.action_planner),
                                    "active": True if monitor.active else False,
                                    "name": "{0}".format(monitor.name),
                                    "obj_to_avoid": "{0}".format(monitor.obj_to_avoid),
                                    "act_to_avoid": "{0}".format(monitor.act_to_avoid),
                                    "rewards": {
                                        "near": float(
                                            "{0:.2f}".format(near)),
                                        "immediate": float(
                                            "{0:.2f}".format(immediate)),
                                        "violated": float(
                                            "{0:.2f}".format(violated)),
                                    }
                                }
                            }
                        if monitor.type in properties_map:
                            properties_map[monitor.type].append(monitor.name)
                        else:
                            properties_map[monitor.type] = [monitor.name]

        if hasattr(elements,"monitors"):
            if hasattr(elements.monitors,"patterns"):
                for type in elements.monitors.patterns:
                    for monitor in type:
                        type_of_monitor = monitor.type
                        respected = 1
                        violated = -1
                        for current_monitor in rewards:
                            if hasattr(current_monitor,"name"):
                                if current_monitor.name == type_of_monitor:
                                    respected = current_monitor.respected
                                    violated = current_monitor.violated
                        list_of_json_patterns[monitor.name] = {
                                "{0}".format(monitor.name): {
                                    "type": "{0}".format(monitor.type),
                                    "mode": "{0}".format(monitor.mode),
                                    "active": True if monitor.active else False,
                                    "name": "{0}".format(monitor.name),
                                    "conditions":"{0}".format(monitor.conditions) if not hasattr(monitor.conditions,"pre") else {
                                        "pre":"{0}".format(monitor.conditions.pre),
                                        "post":"{0}".format(monitor.conditions.post)
                                    },
                                    "rewards": {
                                        "respected": float(
                                             "{0:.2f}".format(respected)),
                                        "violated": float(
                                             "{0:.2f}".format(violated))
                                    }
                                }
                        }
                        if monitor.type in patterns_map:
                            patterns_map[monitor.type].append(monitor.name)
                        else:
                            patterns_map[monitor.type] = [monitor.name]

        json_object = json.dumps({
            "config_name": "randomEnv-{0}x{0}-{1}-v0".format(grid_size, random_token),
            "algorithm": "a2c",
            "env_name": "MiniGrid-RandomEnv-{0}x{0}-{1}-v0".format(grid_size, random_token),
            "num_processes": 48,
            "num_steps": 5,
            "log_interval": 10,
            "on_violation_reset": False,
            "rendering": False,
            "evaluation_directory_name": "evaluations",
            "visdom": False,
            "debug_mode": False,
            "monitors": {
                "properties": {

                },
                "patterns":{

                }
            },
            "rewards": {
                "standard":{
                    "goal": float("{0:.2f}".format(rewards.standard.goal if hasattr(rewards.standard,'goal') else 1)),
                    "step": float("{0:.2f}".format(rewards.standard.step if hasattr(rewards.standard,'step')else 0)),
                    'death': float("{0:.2f}".format(rewards.standard.death if hasattr(rewards.standard,'death') else -1))
                },
                "cleaningenv":{
                    "clean":float("{0:.2f}".format(rewards.cleaningenv.clean if hasattr(rewards.cleaningenv,'clean') else 0.5))
                }
            }
        }, indent=2)

        d = {}
        dProperties = {}
        dPatterns = {}

        for p in properties_map:
            if isinstance(properties_map[p],str):
                if p in dProperties:
                    dProperties[p].update(list_of_json_properties[properties_map[p]])
                else:
                    dProperties[p] = list_of_json_properties[properties_map[p]]
            for value in properties_map[p]:
                if p in dProperties:
                    dProperties[p].update(list_of_json_properties[value])
                else:
                    dProperties[p] = list_of_json_properties[value]
        for p in patterns_map:
            if isinstance(patterns_map[p],str):
                if p in dPatterns:
                    dPatterns[p].update(list_of_json_patterns[patterns_map[p]])
                else:
                    dPatterns[p] = list_of_json_patterns[patterns_map[p]]
            else:
                for value in patterns_map[p]:
                    if p in dPatterns:
                        dPatterns[p].update(list_of_json_patterns[value])
                    else:
                        dPatterns[p] = list_of_json_patterns[value]

        d = json.loads(json_object)
        d['monitors']['properties'].update(dProperties)
        d['monitors']['patterns'].update(dPatterns)
        config.write(json.dumps(d,sort_keys=True,indent=2))
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