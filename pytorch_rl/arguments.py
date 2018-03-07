import argparse

import torch



def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--entropyOffset', type=float, default=0.01,
                        help='entropy term offset (default: 0.01)')
    parser.add_argument('--entropy-Temp', type=float, default=False,
                        help='entropy Temperature coefficient (default: 20 000)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=2,
                        help='how many training CPU processes to use (default: 32)')
    parser.add_argument('--num-steps', type=int, default=1,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--num-stack', type=int, default=1,
                        help='number of frames to stack (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10, #100
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--vis-interval', type=int, default=100, #100
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=10e7,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='MultiRoom-Teacher',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
   
    
    #make an automatic check for the cuda setting on the machine
    useCuda=torch.cuda.is_available()    
    parser.add_argument('--no-cuda', action='store_true', default=useCuda,
                        help='disables CUDA training')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization'),
    parser.add_argument('--useMissionAdvice', type=int, default=1,
                        help='False if not using teacher advices.Else, indicate the number of time steps when the agent uses the advice')
    parser.add_argument('--useActionAdvice', type=int,default=False,
                        help='False if not using teacher best actions.Else, indicate the number of time steps when the agent uses the action')
    parser.add_argument('--load-dir', default='./trained_models/best/',
                    help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--expID', type=int, default=6,
                    help='ID of the experiment to run')
    parser.add_argument('--serverVisdom', default='http://eos11',
                        help='display server used for visdom, default eos11 on elisa2')
    parser.add_argument('--portVisdom', type=int, default=24431,
                    help='ID of the port used for Visdom, default 24431, CHANGE THIS VALUE BEFORE RUNNING THE main.py')
    parser.add_argument('--vizTrain', type=bool, default=False,
                    help='visualize the agent during the training')
    parser.add_argument('--debug', type=bool, default=False,
                    help='active debug mode')
    parser.add_argument('--sentenceEmbeddingDimension', type=int, default=200,
                    help='dimension of the vectors that embedds the missions')
    
    
    
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    if not args.cuda:
        print('*** WARNING: CUDA NOT ENABLED ***')

    return args
