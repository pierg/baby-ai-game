
try:
    import gym_minigrid
    from gym_minigrid.wrappers import *
    from gym_minigrid.envelopes import SafetyEnvelope

except:
    pass

from helpers import config_grabber as cg

def make_env(env_id, seed, rank, log_dir):

    config = cg.Configuration.grab("blocker")

    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)

        env = SafetyEnvelope(env)

        # Maxime: until RL code supports dict observations, squash observations into a flat vector
        if isinstance(env.observation_space, spaces.Dict):
            env = FlatObsWrapper(env)

        return env

    return _thunk
