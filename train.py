#!/usr/bin/env python3

import pytorch_a2c.a2c
from pytorch_a2c.envs import make_env
from pytorch_a2c.arguments import get_args
from pytorch_a2c.vec_env.subproc_vec_env import SubprocVecEnv














def main():

    args = get_args()

    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    """
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
    """



    envs = [make_env(args.env_name, args.seed, i) for i in range(args.num_processes)]
    envs = SubprocVecEnv(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    obs_numel = reduce(operator.mul, obs_shape, 1)

    assert envs.action_space.__class__.__name__ == "Discrete"
    action_shape = 1

    actor_critic = Policy(obs_numel, envs.action_space)




    """
        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model, hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))
    """

if __name__ == "__main__":
    main()
