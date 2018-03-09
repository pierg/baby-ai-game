import argparse
import os
import sys
import types
import time

import numpy as np
import torch
from torch.autograd import Variable
from vec_env.dummy_vec_env import DummyVecEnv
import preProcess

from envs import make_env
from arguments import get_args
args = get_args()


env = make_env(args.env_name, args.seed, 0)
env = DummyVecEnv([env])

actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.nameExpToLoad, args.env_name + ".pt"))

render_func = env.envs[0].render

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

current_obs = torch.zeros(1, *obs_shape)

maxSizeOfMissionsSelected=args.sentenceEmbeddingDimension
preProcessor=preProcess.PreProcessor(maxSizeOfMissionsSelected)
current_missions=torch.zeros(1, maxSizeOfMissionsSelected)

states = torch.zeros(1, actor_critic.state_size)
masks = torch.zeros(1, 1)

#def update_current_obs(obs):
#    obs=obs[0]['image']
#    shape_dim0 = env.observation_space.shape[0]
#    obs = torch.from_numpy(obs).float()
#    if args.num_stack > 1:
#        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
#    current_obs[:, -shape_dim0:] = obs

def update_current_obs(obs,missions):
        #print('top')
        shape_dim0 = env.observation_space.shape[0]
        #img,txt = torch.from_numpy(np.stack(obs[:,0])).float(),np.stack(obs[:,1])

        images = torch.from_numpy(obs)
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = images
        current_missions = missions

render_func('human')


obsF = env.reset()
obs=np.array([preProcessor.preProcessImage(dico['image']) for dico in obsF])
missions=torch.stack([preProcessor.stringEncoder('go to the green square') for dico in obsF])
    
    
update_current_obs(obs,missions)


step=0
allowObserveReward=False
  
while True:
    useAdviceFromTeacher=False
    if not args.useMissionAdvice == False:
        if step%args.useMissionAdvice==0:
            useAdviceFromTeacher=True    
    step+=1
    
       
    
    
    
    #preprocess the missions to be used by the model
    missionsAsStrings=preProcessor.Code2String(missions)
    print('mission',missionsAsStrings)
    #print('missions', missionsAsStrings)
    #then use the language model of our choice
    missionsEmbedded=preProcessor.simpleSentenceEmbedding(missionsAsStrings)
    print('embedding:', missionsEmbedded)
    #print('   ')
    #convert them as Variables
    missionsVariable=Variable(missionsEmbedded,volatile=True)
       
        
                
    value, action, _, states = actor_critic.act(
        Variable(current_obs, volatile=True),
        Variable(states, volatile=True),
        Variable(masks, volatile=True),
        deterministic=True,
        missions=missionsVariable
    )
    states = states.data
    cpu_actions = action.data.squeeze(1).cpu().numpy()

    # Observation, reward and next obs
    obsF, reward, done, _ = env.step(cpu_actions,allowObserveReward)
    obs=np.array([preProcessor.preProcessImage(dico['image']) for dico in obsF])
    
#    if useAdviceFromTeacher:
#        missions=torch.stack([preProcessor.stringEncoder(dico['mission']) for dico in obsF])
#    else:
#        missions=torch.stack([preProcessor.stringEncoder('go to the goal') for dico in obsF])
    missions=torch.stack([preProcessor.stringEncoder('go to the green square') for dico in obsF])

    bestActions=Variable(torch.stack( [ torch.Tensor(dico['bestActions']) for dico in obsF ] ))
            

    #time.sleep(0.05)

    masks.fill_(0.0 if done else 1.0)

    if current_obs.dim() == 4:
        current_obs *= masks.unsqueeze(2).unsqueeze(2)
    else:
        current_obs *= masks
    current_missions *= masks

    update_current_obs(obs,missions)

    renderer = render_func('human')

    if not renderer.window:
        sys.exit(0)
