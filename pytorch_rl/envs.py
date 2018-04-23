
try:
    import gym_minigrid
    from gym_minigrid.wrappers import *
    from gym_minigrid.envelopes import SafetyEnvelope

except:
    pass

from helpers import config_grabber as cg

def make_env(env_id, seed, rank, log_dir, reset_on_catastrophe=False):

    config = cg.Configuration.grab()

    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)

        if config.blocker:
            env = SafetyEnvelope(env, reset_on_catastrophe=reset_on_catastrophe)

        # Maxime: until RL code supports dict observations, squash observations into a flat vector
        if isinstance(env.observation_space, spaces.Dict):
            env = FlatObsWrapper(env)

        return env

    return _thunk
