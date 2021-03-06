# Baby AI Game

[![Build Status](https://travis-ci.org/maximecb/baby-ai-game.svg?branch=master)](https://travis-ci.org/maximecb/baby-ai-game)

Prototype of a game where a reinforcement learning agent is trained through natural language instructions. This is a research project based at the [Montreal Institute for Learning Algorithms (MILA)](https://mila.quebec/en/).

## Installation

Requirements:
- Python 3.5+
- OpenAI gym
- NumPy
- PyQT5
- PyTorch

Start by manually installing PyTorch. See the [PyTorch website](http://pytorch.org/)
for installation instructions specific to your platform.

Then, install the [minigrid Gym environment](https://github.com/maximecb/gym-minigrid):

```
git clone https://github.com/pierg/gym-minigrid.git
cd gym-minigrid
pip3 install -e .
cd ..
```

Then, clone this repository and install the other dependencies with `pip3`:

```
git clone https://github.com/pierg/baby-ai-game.git
cd baby-ai-game
pip3 install -e .
```

### For conda users

If you are using conda, you can create a `babyai` environment with all the dependencies by running:

```
conda env create -f environment.yaml
```

Having done that, you can either add `baby-ai-game` and `gym-minigrid` in your `$PYTHONPATH` or install them in the development mode as suggested above.

## Structure of the Codebase

The `levels` directory will contain all the code relevant to the generation of levels and missions. Essentially, this implements the test task for the Baby AI Game. This is an importable module which people can use on its own to perform experiments.

The `agents` directory will contain a default implementation of one or more agents to be evaluated on the baby AI game. This should also be importable as an independent module. Each agent will need to support methods to be provided teaching inputs using pointing and naming, as well as demonstrations.

In `pytorch_rl`, there is an implementation of the A2C, PPO and ACKTR reinforcement learning algorithms. This is a custom fork of [this repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) which has been adapted to work with the `gym-minigrid` environment. This RL implementation has issues and will hopefully be replaced by a better one soon. One important problem, for instance, is that it is not importable as a module.

The `main.py` script implements a template of a user interface for interactive human teaching. The version found in the master branch allows you to control the agent manually with the arrow keys, but it is not currently connected to any model or teaching code. Currently, most experiments are done offline, without a user interface.

## Instructions for Committers

To contribute to this project, you should first create your own fork, and remember to periodically [sync changes from this repository](https://stackoverflow.com/questions/7244321/how-do-i-update-a-github-forked-repository). You can then create [pull requests](https://yangsu.github.io/pull-request-tutorial/) for modifications you have made. Your changes will be tested and reviewed before they are merged into this repository. If you are not familiar with forks and pull requests, I recommend doing a Google or YouTube search to find many useful tutorials on the issue. Knowing how to use git and GitHub effectively are valuable skills for any programmer.

If you have found a bug, or would like to request a change or improvement
to the grid world environment or user interface, please
[open an issue](https://github.com/maximecb/baby-ai-game/issues)
on this repository. For bug reports, please paste complete error messages and describe your system configuration (are you running on Mac, Linux?).

## Usage

You can run the example scripts `start_training.sh` and `start_results.sh` specifying the name of the environment:

Ex:
```
MiniGrid-Empty-6x6-v0
MiniGrid-Empty-8x8-v0
MiniGrid-Empty-16x16-v0
MiniGrid-DoorKey-5x5-v0
MiniGrid-DoorKey-6x6-v0
MiniGrid-DoorKey-8x8-v0
MiniGrid-DoorKey-16x16-v0
MiniGrid-MultiRoom-N2-S4-v0
MiniGrid-MultiRoom-N6-v0
MiniGrid-Fetch-5x5-N2-v0
MiniGrid-Fetch-6x6-N2-v0
MiniGrid-Fetch-8x8-N3-v0
MiniGrid-GoToDoor-5x5-v0
MiniGrid-GoToDoor-6x6-v0
MiniGrid-GoToDoor-8x8-v0
MiniGrid-PutNear-6x6-N2-v0
MiniGrid-PutNear-8x8-N3-v0
MiniGrid-LockedRoom-v0
MiniGrid-FourRoomQA-v0
```

To run the interactive UI application:

```
./main.py
```

To run the environments without the script first export the `gym_minigrid` and `gym_minigrid/envs/` to PYTHONPATH
```
PYTHONPATH=./gym_minigrid/:$PYTHONPATH
PYTHONPATH=./gym_minigrid/envs/:$PYTHONPATH
export PYTHONPATH
```

The environment being run can be selected with the `--env-name` option, eg:

```
./main.py --env-name MiniGrid-Fetch-8x8-N3-v0
```

Basic reinforcement learning code is provided in the `pytorch_rl` subdirectory.
You can perform training using the A2C algorithm with:

```
python3 pytorch_rl/main.py --env-name MiniGrid-Empty-6x6-v0 --no-vis --num-processes 48 --algo a2c
```

In order to Use the teacher environment with pytorch_rl, use the following command :
```
python3 pytorch_rl/main.py --env-name MultiRoom-Teacher --no-vis --num-processes 48 --algo a2c
```

Note: the pytorch_rl code is a custom fork of [this repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr),
which was modified to work with this environment.

To see the available environments and their implementation, please have a look at
the [gym_minigrid](https://github.com/maximecb/gym-minigrid) repository.



## Usage of the Environments

To run the standalone UI application, which allows you to manually control the agent with the arrow keys:

```
./standalone.py
```

The environment being run can be selected with the `--env-name` option, eg:

```
./standalone.py --env-name MiniGrid-Empty-8x8-v0
```

Basic reinforcement learning code is provided in the `pytorch_rl` subdirectory.
You can perform training using the A2C algorithm with:

```
python3 pytorch_rl/main.py --env-name MiniGrid-Empty-6x6-v0 --no-vis --num-processes 48 --algo a2c
```

You can view the result of training using the `enjoy.py` script:

```
python3 pytorch_rl/enjoy.py --env-name MiniGrid-Empty-6x6-v0 --load-dir ./trained_models/a2c
```

## Design

MiniGrid is built to support tasks involving natural language and sparse rewards.
The observations are dictionaries, with an 'image' field, partially observable
view of the environment, and a 'mission' field which is a textual string
describing the objective the agent should reach to get a reward. Using
dictionaries makes it easy for you to add additional information to observations
if you need to, without having to force everything into a single tensor.
If your RL code expects a tensor for observations, please take a look at
`FlatObsWrapper` in
[gym_minigrid/wrappers.py](/gym_minigrid/wrappers.py).

The partially observable view of the environment uses a compact and efficient
encoding, with just 3 input values per visible grid cell, 147 values total.
If you want to obtain an array of RGB pixels instead, see the `getObsRender` method in
[gym_minigrid/minigrid.py](gym_minigrid/minigrid.py).

Structure of the world:
- The world is an NxM grid of tiles
- Each tile in the grid world contains zero or one object
  - Cells that do not contain an object have the value `None`
- Each object has an associated discrete color (string)
- Each object has an associated type (string)
  - Provided object types are: wall, door, locked_doors, key, ball, box and goal
- The agent can pick up and carry exactly one object (eg: ball or key)

Actions in the basic environment:
- Turn left
- Turn right
- Move forward
- Toggle (pick up or interact with objects)
- Wait (noop, do nothing)

By default, sparse rewards for reaching a goal square are provided, but you can
define your own reward function by creating a class derived from MiniGridEnv. Extending
the environment with new object types or action should be very easy very easy.
If you wish to do this, you should take a look at the
[gym_minigrid/minigrid.py](gym_minigrid/minigrid.py) source file.

## Included Environments

The environments listed below are implemented in the [gym_minigrid/envs](/gym_minigrid/envs) directory.
Each environment provides one or more configurations registered with OpenAI gym. Each environment
is also programmatically tunable in terms of size/complexity, which is useful for curriculum learning
or to fine-tune difficulty.

### Empty environment

Registered configurations:
- `MiniGrid-Empty-6x6-v0`
- `MiniGrid-Empty-8x8-v0`
- `MiniGrid-Empty-16x16-v0`

<p align="center">
<img src="/figures/empty-env.png" width=250>
</p>

This environment is an empty room, and the goal of the agent is to reach the
green goal square, which provides a sparse reward. A small penalty is
subtracted for the number of steps to reach the goal. This environment is
useful, with small rooms, to validate that your RL algorithm works correctly,
and with large rooms to experiment with sparse rewards.

### Door & key environment

Registered configurations:
- `MiniGrid-DoorKey-5x5-v0`
- `MiniGrid-DoorKey-6x6-v0`
- `MiniGrid-DoorKey-8x8-v0`
- `MiniGrid-DoorKey-16x16-v0`

<p align="center">
<img src="/figures/door-key-env.png">
</p>

This environment has a key that the agent must pick up in order to unlock
a goal and then get to the green goal square. This environment is difficult,
because of the sparse reward, to solve using classical RL algorithms. It is
useful to experiment with curiosity or curriculum learning.

### Multi-room environment

Registered configurations:
- `MiniGrid-MultiRoom-N2-S4-v0` (two small rooms)
- `MiniGrid-MultiRoom-N6-v0` (six room)

<p align="center">
<img src="/figures/multi-room.gif" width=416 height=424>
</p>

This environment has a series of connected rooms with doors that must be
opened in order to get to the next room. The final room has the green goal
square the agent must get to. This environment is extremely difficult to
solve using classical RL. However, by gradually increasing the number of
rooms and building a curriculum, the environment can be solved.

### Fetch environment

Registered configurations:
- `MiniGrid-Fetch-5x5-N2-v0`
- `MiniGrid-Fetch-6x6-N2-v0`
- `MiniGrid-Fetch-8x8-N3-v0`

<p align="center">
<img src="/figures/fetch-env.png" width=450>
</p>

This environment has multiple objects of assorted types and colors. The
agent receives a textual string as part of its observation telling it
which object to pick up. Picking up the wrong object produces a negative
reward.

### Go-to-door environment

Registered configurations:
- `MiniGrid-GoToDoor-5x5-v0`
- `MiniGrid-GoToDoor-6x6-v0`
- `MiniGrid-GoToDoor-8x8-v0`

<p align="center">
<img src="/figures/gotodoor-6x6.png" width=400>
</p>

This environment is a room with four doors, one on each wall. The agent
receives a textual (mission) string as input, telling it which door to go to,
(eg: "go to the red door"). It receives a positive reward for performing the
`wait` action next to the correct door, as indicated in the mission string.

### Put-near environment

Registered configurations:
- `MiniGrid-PutNear-6x6-N2-v0`
- `MiniGrid-PutNear-8x8-N3-v0`

The agent is instructed through a textual string to pick up an object and
place it next to another object. This environment is easy to solve with two
objects, but difficult to solve with more, as it involves both textual
understanding and spatial reasoning involving multiple objects.

### Locked Room Environment

Registed configurations:
- `MiniGrid-LockedRoom-v0`

The environment has six rooms, one of which is locked. The agent receives
a textual mission string as input, telling it which room to go to in order
to get the key that opens the locked room. It then has to go into the locked
room in order to reach the final goal. This environment is extremely difficult
to solve with vanilla reinforcement learning alone.

### Four room question answering environment

Registered configurations:
- `MiniGrid-FourRoomQA-v0`

<p align="center">
<img src="/figures/fourroomqa-env.png">
</p>

This environment is inspired by the
[Embodied Question Answering](https://arxiv.org/abs/1711.11543) paper. The question are of the form:

> Are there any keys in the red room?

There are four colored rooms, and the agent starts at a random position in the grid.
Multiple objects of different types and colors are also placed at random
positions in random rooms. A question and answer pair is generated, the
question is given to the agent as an observation, and the agent has a limited
number of time steps to explore the environment and produce a response. This
environment can be easily modified to add more question types or to diversify
the way the questions are phrased.



### Usage at MILA

If you connect to the lab machines by ssh-ing, make sure to use `ssh -X` in order to see the game window. This will work even for a chain of ssh connections, as long as you use `ssh -X` at all intermediate steps. If you use screen, set `$DISPLAY` variable manually inside each of your screen terminals. You can find the right value for `$DISPLAY` by detaching from you screen first (`Ctrl+A+D`) and then running `echo $DISPLAY`.

The code does not work in conda, install everything with `pip install --user`.

### Troubleshooting

If you run into error messages relating to OpenAI gym or PyQT, it may be that the version of those libraries that you have installed is incompatible. You can try upgrading specific libraries with pip3, eg: `pip3 install --upgrade gym`. If the problem persists, please [open an issue](https://github.com/maximecb/baby-ai-game/issues) on this repository and paste a *complete* error message, along with some information about your platform (are you running Windows, Mac, Linux? Are you running this on a MILA machine?).

## About this Project

The Baby AI Game is a game in which an agent existing in a simulated world
will be trained to complete task through reinforcement learning as well
as interactions from one or more human teachers. These interactions will take
the form of natural language, and possibly other feedback, such as human
teachers manually giving rewards to the agent, or pointing towards
specific objects in the game using the mouse.

Two of the main goals of the project are to explore ways in which deep learning can take
inspiration from human learning (ie: how human babies learn), and to research AI learning
with humans in the loop. In particular, language learning,
as well as teaching agents to complete actions spanning many (eg: hundreds)
of time steps, or macro-actions composed of multiple micro-actions, are
still open research problems.

Some possible approaches to be explored in this project include meta-learning
and curriculum learning, the use of intrinsic motivation (curiosity), and
the use of pretraining to give agents a small core of built-in knowledge to
allow them to learn from human agents. With respect to built-in knowledge,
Yoshua Bengio believes that the ability for agents to understand pointing
gestures in combination with language may be key.

You can find here a presentation of the project: [Baby AI Summary](https://docs.google.com/document/d/1WXY0HLHizxuZl0GMGY0j3FEqLaK1oX-66v-4PyZIvdU)

A work-in-progress review of related work can be found [here](https://www.overleaf.com/13480997qqsxybgstxhg#/52042269/)
