import copy
import glob
import os
import time
import operator
from functools import reduce

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import json

from arguments import get_args
from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from kfac import KFACOptimizer
from model import RecMLPPolicy, MLPPolicy, CNNPolicy,easyPolicy
from storage import RolloutStorage
from visualize import visdom_plot
import preProcess
import pickle

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], 'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#try:
#    os.makedirs(args.log_dir)
#except OSError:
#    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
#    for f in files:
#        os.remove(f)




def main():
    
    
# =============================================================================
#     Create an appropriate folder to store the experiment, info...
# =============================================================================
    experimentNumber=0
    if not args.expName is False:
        experimentFolder=args.expName+'Exp{}'.format(experimentNumber)
    else:
        experimentFolder='Exp{}'.format(experimentNumber)
    
    save_path = os.path.join(args.save_dir, args.algo,experimentFolder)
    while os.path.exists(save_path):
        print('previous experiment ID used : ', experimentNumber)
        experimentNumber+=1
        if not args.expName is False:
             experimentFolder=args.expName+'Exp{}'.format(experimentNumber)
        else:
            experimentFolder='Exp{}'.format(experimentNumber)
        save_path = os.path.join(args.save_dir, args.algo,experimentFolder)
        
        
    print('saving results in ',save_path)
    os.makedirs(save_path)
        
    #number of times the agent has won all the envs 
    numberOfSuccess=0
    
    #save the number of interactions required to solve the env
    interactionCounter=0

# =============================================================================
#   Info about the experiment that wil be saved and plotted
# =============================================================================
    infoToSave={'timestep':[],
          'FPS':[],
          'meanReward':[],
          'medianReward':[],
          'minReward':[],
          'minReward':[],
          'maxReward':[],
          'entropy':[],
          'valueLoss':[],
          'actionLoss':[],
          #count the number of times an action has been selected by the agent or the teacher
          'numberOfChoices_Teacher':[], 
          'numberOfChoices_Agent':[],
          #save the ratio of these previous numbers. Redundant but easier to track
          'actionRatio':[],
          'entropyCoef':[],
          'envFinished':[]}   
                    
          
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'
    
# =============================================================================
#     Generate a descriptor of the current experiment
# =============================================================================
    descriptor=''
    descriptor+='Experiment {} \n'.format(experimentFolder)
    descriptor+="experience done on : {} at {}  \n".format(time.strftime("%d/%m/%Y"),time.strftime("%H:%M:%S"))
    
    for i in vars(args):
            line_new = '{:>12}  {:>12} \n'.format(i, getattr(args,i))
            descriptor+=line_new
    fileInfo=os.path.join(save_path,'summary.txt')
    f= open(fileInfo,"w")
    f.write(descriptor)
    f.close()

# =============================================================================
#   Create Visdom environment
# =============================================================================
    if args.vis:
        from visdom import Visdom
        print('using VISDOM')
        #specific to local experiments on windows laptop [Simon]
        if os.name=='nt':
            print('using visdom for testing on local windows machine')
            viz = Visdom(env=experimentFolder,port=8097)

        else:
            print('using visdom on a linux server')
            viz = Visdom(server=args.serverVisdom,port=args.portVisdom,env=experimentFolder)
        
        
        #plot a summary of the experiment on visdom
        viz.text(descriptor)
        #windows that will be plotted on visdom
        win = {'rewards':None,
               'entropy':None,
               'statsAction':None,
               'actionRatio':None,
               'entropyCoef':None,
               'envFinished':None}


# =============================================================================
#   Create the different environments for the parallel training
# =============================================================================
    envs = [make_env(args.env_name, args.seed, i)
                for i in range(args.num_processes)]
    

    # dictionnary that describes the name of the actions in the environment
    # it is represented as {'name of action':actionID}
    actionDescription=False
    
# =============================================================================
#   embedds the envs in order to share them with the threads
# =============================================================================
    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    
# =============================================================================
#   decide wether to plot the render of the env at each time step or not
# =============================================================================
    if args.vizTrain:
        render_func = envs.envs[0].render

    # Maxime: commented this out because it very much changes the behavior
    # of the code for seemingly arbitrary reasons
    #if len(envs.observation_space.shape) == 1:
    #    envs = VecNormalize(envs)




    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    obs_numel = reduce(operator.mul, obs_shape, 1)
    
# =============================================================================
#     select the arcitecture for the actor-critic
# =============================================================================
    architecture='None'

    if len(obs_shape) == 3 and obs_numel > 1024:
        actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy)
        architecture='CNN policy'
    elif args.recurrent_policy:
        actor_critic = RecMLPPolicy(obs_numel, envs.action_space)
        architecture='Reccurent policy'
    else:
        actor_critic = MLPPolicy(obs_numel, envs.action_space)
        architecture='MLP policy'
        
    
# =============================================================================
#     DEBUG MODE
# =============================================================================
    if args.debug:
        print('using easy policy for debug')
        actor_critic=easyPolicy(obs_numel, envs.action_space)
        architecture='debug policy'
        
    #log the architecture info
    print(architecture)
    f= open(fileInfo,"a")
    newLine='{:>12}  {:>12} \n'.format('architecture', architecture)
    f.write(newLine)
    f.close()
    
    #useful to compute the 'envFinished' counter
    current_envFinished=np.zeros((args.num_processes))
    
    #initialize the ratio counter
    numberOfActions=envs.action_space.n
    #print('before',  infoToSave['actionRatio'])
    infoToSave['actionRatio']=[[] for i in range(numberOfActions)]
    #print('after',  infoToSave['actionRatio'])
    
    
    # Maxime: log some info about the model and its size
    modelSize = 0
    for p in actor_critic.parameters():
        pSize = reduce(operator.mul, p.size(), 1)
        modelSize += pSize
    print(str(actor_critic))
    print('Total model size: %d' % modelSize)


    #select the action space
    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]
    
    #choose wether to use 
    if args.cuda:
        actor_critic.cuda()

    #select RL algorithm
    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    elif args.algo == 'ppo':
        optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)
    elif args.algo == 'acktr':
        optimizer = KFACOptimizer(actor_critic)


    #the missions are embedded into vectors of size :
    maxSizeOfMissionsSelected=args.sentenceEmbeddingDimension
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size,maxSizeOfMissions=maxSizeOfMissionsSelected)
    
    #setup the current observations
    current_obs = torch.zeros(args.num_processes, *obs_shape)
    
# =============================================================================
#     The Preprocessor allows to convert strings into ASCII codes that will be stored
#   it also pre-process the images when the environment is not suited to this framweork
#   it also computes sentence embeddings for the missions given as strings
# =============================================================================
    preProcessor=preProcess.PreProcessor(maxSizeOfMissionsSelected)
    
    #update the current missions from the teacher
    current_missions=torch.zeros(args.num_processes, maxSizeOfMissionsSelected)

    #temp variable used to compute the info['numberOfChoices...']
    currentCount={'numberOfChoices_Teacher':[0 for i in range(numberOfActions)],
                                             'numberOfChoices_Agent':[0 for i in range(numberOfActions)]}

    
    #as the training goes, the current_obs is updated
    #take care, here obs means image, and obsFull={image, mission, bestActionID}
    def update_current_obs(obs,missions):
        #global current_missions
        #print('top')
        shape_dim0 = envs.observation_space.shape[0]
        #img,txt = torch.from_numpy(np.stack(obs[:,0])).float(),np.stack(obs[:,1])

        images = torch.from_numpy(obs)
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = images
        current_missions[:,:] = missions


    obsF = envs.reset()
#    print('init')
#    print(obs)

    #print('obs : ', obs)
#    print(len(obs))
#    print(obs[0])
    
    #obsF,reward,done,info=envs.step(np.ones((args.num_processes)))
    #print('after 1 step')
    #print(obs)
    
    #reshape the image, if needed
    obs=np.array([preProcessor.preProcessImage(dico['image']) for dico in obsF])
    
    
    
    
    #translate the string missions in vector of ASCII code
    #missions=torch.stack([preProcessor.stringEncoder(dico['mission']) for dico in obsF])
    missions=torch.stack([preProcessor.stringEncoder('go to the green square') for dico in obsF])
    #id of the best actions computed by the teacher, they are not given to the agent,
    #but they are used to compute action statistics
    bestActions=[dico['bestActions'] for dico in obsF ] 

    
    #bestActions=Variable(torch.stack( [ torch.Tensor(dico['bestActions']) for dico in obsF ] ))
    #print(missions)
    #print('missions size',missions.size())
    #print(len(obs[0]))
    #print(obs)
    
    
    def getMissionsAsVariables(step,end=False,volatile=False):
        '''
        Allow to convert from list of ASCII codes to pytorch Variables
        the argument step allows point-wise selection in the rollout
        the argument end allows to access a whole part of the memory according to
        missions[step:end]
        '''
        

        #get the missions as a minibatch
        if end is not False:
            tmpMissions=rollouts.missions[step:end].view(-1,maxSizeOfMissionsSelected)
               
        else:
            tmpMissions=rollouts.missions[step]#.view(-1,maxSizeOfMissionsSelected)
        
        #here tmp missions is a list of ASCII codes
        #for i in range(tmpMissions.size()[0]):
        #    print('tmp missions', tmpMissions[i,0:10])
        #we get the original strings
        missionsAsStrings=preProcessor.Code2String(tmpMissions)
        #print('missions', missionsAsStrings)
        #then use the language model of our choice
        missionsEmbedded=preProcessor.simpleSentenceEmbedding(missionsAsStrings)
        #print('   ')
        #convert them as Variables
        missionsVariable=Variable(missionsEmbedded,volatile=volatile)
       
      
        
        #check if cuda is available
        if args.cuda:
            missionsVariable=missionsVariable.cuda()
        return(missionsVariable)
        
        
    
    def correctReward(reward, cpu_actions,cpu_teaching_actions):
        '''
        defines the correction on the reward to apply in order to take account of the fact
        that in mode teacher the agent might choose wrong actions while actually 
        applying right actions, because actions are overwriten in the teacher mode
        '''
        si=len(cpu_actions)
        output=0
        #print('chosen ',cpu_actions)
        #print('teaching ',cpu_teaching_actions)
        #print('reward', reward)
        for i in range(si):
            if int(cpu_actions[i]) != int (cpu_teaching_actions[i]):
                reward[i]-=2
        return(output)
        
        
    def updateNumberOfActions(currentCount, actions_Agent, actions_Teacher):
        '''
        This function is used to keep track of the actions selected by the agent
        it updates the number of times that a certain action has been selected by the agent
        and the number of times an action has been indicated by the teacher
        
        currentCount : array of size numberOfActions, 
        actions_Agent/Teacher : array containing the actions id selected by the agent/teacher. 
                                Size Nenvs*WhateverNumberOfPossibleActions
        '''
        
                
        for envAction in actions_Agent:
            for actionID in envAction:
                currentCount['numberOfChoices_Agent'][int(actionID)]+=1
            
        for envAction in actions_Teacher:
            for actionID in envAction:
                currentCount['numberOfChoices_Teacher'][int(actionID)]+=1
        
        
        return(0)
    
    def updateRatioActions(currentRatio,actions_Agent, actions_Teacher):
        '''
        update the info['ratio'] value, using the previously computed 
        numberOfActionsTeacher/Agent
        
        if numberOFActionsAgent=0, then ratio=-1
        '''
        #print(currentRatio)
        for indexAction in range(numberOfActions):
            if actions_Teacher[indexAction]!=0:
                currentRatio[indexAction]+=[actions_Agent[indexAction]/actions_Teacher[indexAction]]
            else:
                currentRatio[indexAction]+=[-1]
        
            
    def forceTeacherMissions(bestActions):
        '''
        select the best action to take, among a pool of optimal actions computed by the teacher
        '''
        output=[]
        for envActions in bestActions:
            value=np.random.choice(envActions)
            output+=[value]
        return(output)
        
        
        
        
    
#    
#    
    #envs.getText()
    #print(txt)
    
    #update the current image and missions as observations
    update_current_obs(obs,missions)
    
    
    
   # for i in range(2):
    #    print('missions', missions[i,0:10])
        
    #for i in range(2):
    #    print('current_mission', current_missions[i,0:10])
    
    rollouts.observations[0].copy_(current_obs)
    
    
    
    
    rollouts.missions[0].copy_(current_missions)
    #print('from rollout', rollouts.missions[0,0,0:10])
    #print('from rollout', rollouts.missions[0,1,0:10])
    #time.sleep(100)
    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
    
    # allow to keep track of the model which had the best mean reward
    bestMeanRewards=final_rewards.mean()
    
    if args.cuda:
        current_obs = current_obs.cuda()
        current_missions=current_missions.cuda()
        #bestActions=bestActions.cuda()
        rollouts.cuda()

    start = time.time()
    
    #coefficient used in the computation of the entropy term, in the loss function
    entropy_offset=args.entropyOffset
    
    useMissionAdvice=args.useMissionAdvice
    
    entropyTime=0
# =============================================================================
#   Beginning of the training
# =============================================================================
    for j in range(num_updates):
        #performs a rollout of n steps
        for step in range(args.num_steps):            
            
            #compute the entropy coefficient for this timestep
            #two posisbility, with or without annealing
            totalTimeStep=(j + 1) * args.num_processes * args.num_steps
            #print(step)

            if not args.entropy_Temp is False:
                #print('using entropy Annealing : 'args.entropy_Temp)
                entropy_coef=entropy_offset + np.exp(-entropyTime/args.entropy_Temp)
                entropyTime+=args.num_processes
            else:
               entropy_coef=args.entropy_coef
         
            
            
# =============================================================================
#         here we define at which frequency we want to hear the advice of the teacher
#           and at which frequency we allow the teacher to take control of the agent 
#           in order to perform the optimal action
# =============================================================================
            #observe the reward form the teacher, +-1 if perform right action
            observeReward=False
            
            useAdviceFromTeacher=False
            if not useMissionAdvice == False:
                if step%useMissionAdvice==0:
                    useAdviceFromTeacher=True   
                    observeReward=True
                    interactionCounter+=args.num_processes
                    #print(interactionCounter)
            
            
            useMissionFromTeacher=False  
            #print('argument useActionAdvice :',args.useActionAdvice)
            if not args.useActionAdvice == False:
                if step%args.useActionAdvice==0:
                    useMissionFromTeacher=True 
                    interactionCounter+=args.num_processes
                    #print(interactionCounter)
                    
            
                   
            
            
            missionsVariable=getMissionsAsVariables(step,volatile=True)

            #preprocess the missions to be used by the model
#            if useAdviceFromTeacher:
#                missionsVariable=getMissionsAsVariables(step,volatile=True)
#            else:
#                missionsVariable=False
            
            
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(
                Variable(rollouts.observations[step], volatile=True),
                Variable(rollouts.states[step], volatile=True),
                Variable(rollouts.masks[step], volatile=True),
                missions=missionsVariable
            )
            
            #keep track of the numberOfActionAgent/Teacher
            updateNumberOfActions(currentCount, action.data, bestActions)
            
            #put the actions in a convenient format
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            
            
                       
            #print('   ')
            #If the teacher takes control of the agent, we use the actions given
            if useMissionFromTeacher:
                cpu_teaching_actions=forceTeacherMissions(bestActions)
                #print('from main : observe reward True')
                obsF, reward, done, info = envs.step(cpu_teaching_actions,observeReward=True)
                #correctReward(reward,cpu_actions,cpu_teaching_actions)
                #print('corrected reward', reward)
            else:
                #normal case, we use the actions computed by the agent
                #print('form main, observe reward',observeReward)
                obsF, reward, done, info = envs.step(cpu_actions,observeReward=observeReward)
            #print('   ')
            #We require the env to give a description of the actions in the info dic
            #Once we have gathered these info, this step is not mandatory anymore
            if actionDescription is False:
                actionDescription=info[0]['actionDescription']
                
                
            ## get the image and mission observation from the observation dictionnary
            obs=np.array([preProcessor.preProcessImage(dico['image']) for dico in obsF])
#            if useAdviceFromTeacher:
#                missions=torch.stack([preProcessor.stringEncoder(dico['mission']) for dico in obsF])
#            else:
#                missions=torch.stack([preProcessor.stringEncoder('go to the goal') for dico in obsF])
            missions=torch.stack([preProcessor.stringEncoder('go to the green square') for dico in obsF])

            
            #bestActions=Variable(torch.stack( [ torch.Tensor(dico['bestActions']) for dico in obsF ] ))
            bestActions=[dico['bestActions'] for dico in obsF ] 

            
            
            #here the rewards are given as [rewardFromThread1, rewardFromThread2,...]
            #we convert them in a pytorch tensor
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            
            #keep track of the reward along an episode
            episode_rewards += reward

            # If done then clean the history of observations
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            #while a thread i is not done, masks[i]=1
            
            #check which thread has finished
            for i, done_ in enumerate(done):
                if done_:
                    if info[i]['finished']==True:
                        current_envFinished[i]=1
                    else:
                        current_envFinished[i]=0
                
            
            
            final_rewards *= masks
            
            #if thread i is done, final_reward[i]=episode_rewqrd
            #else, final_reward[i]=0
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks
            current_missions *= masks

                
            #update current observation and save it in the storage memory
            update_current_obs(obs,missions)
            rollouts.insert(step, current_obs, current_missions, states.data, action.data, action_log_prob.data, value.data, reward, masks)


            missionsVariable=getMissionsAsVariables(-1,volatile=True)

#        if useAdviceFromTeacher:
#            missionsVariable=getMissionsAsVariables(-1,volatile=True)
#        else:
#            missionsVariable=False
            
        next_value = actor_critic(
            Variable(rollouts.observations[-1], volatile=True),
            Variable(rollouts.states[-1], volatile=True),
            Variable(rollouts.masks[-1], volatile=True),
            missions=missionsVariable
        )[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        if args.algo in ['a2c', 'acktr']:
            
            missionsVariable=getMissionsAsVariables(0,end=-1,volatile=False)
            values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
                Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                Variable(rollouts.states[:-1].view(-1, actor_critic.state_size)),
                Variable(rollouts.masks[:-1].view(-1, 1)),
                Variable(rollouts.actions.view(-1, action_shape)),
                missions=missionsVariable
            )

            values = values.view(args.num_steps, args.num_processes, 1)
            action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages.data) * action_log_probs).mean()

            if args.algo == 'acktr' and optimizer.steps % optimizer.Ts == 0:
                # Sampled fisher, see Martens 2014
                actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = Variable(torch.randn(values.size()))
                if args.cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - Variable(sample_values.data)).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                optimizer.acc_stats = False

            optimizer.zero_grad()
            (value_loss * args.value_loss_coef + action_loss - dist_entropy * entropy_coef).backward()

            ## CLIP THE GRADIENT 
            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()
        elif args.algo == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for e in range(args.ppo_epoch):
                if args.recurrent_policy:
                    data_generator = rollouts.recurrent_generator(advantages, args.num_mini_batch)
                else:
                    data_generator = rollouts.feed_forward_generator(advantages, args.num_mini_batch)

                for sample in data_generator:
                    observations_batch, missions_batch, states_batch, actions_batch, \
                       return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(
                        Variable(observations_batch),
                        Variable(states_batch),
                        Variable(masks_batch),
                        Variable(actions_batch)
                    )

                    adv_targ = Variable(adv_targ)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                    value_loss = (Variable(return_batch) - values).pow(2).mean()

                    optimizer.zero_grad()
                    (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
                    nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
                    optimizer.step()

        rollouts.after_update()

        if args.vizTrain:
            renderer = render_func('human')

        if j % args.save_interval == 0 and args.save_dir != "":
            #print('current advice',envs.s)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            if bestMeanRewards<final_rewards.mean():
                print('updating best model to save...')
                print('Previous best mean reward : ',bestMeanRewards, 'new : ',final_rewards.mean())
                
                bestMeanRewards=final_rewards.mean()
                save_model = actor_critic
                if args.cuda:
                    save_model = copy.deepcopy(actor_critic).cpu()
    
                save_model = [save_model,
                                hasattr(envs, 'ob_rms') and envs.ob_rms or None]
                torch.save(save_model, os.path.join(save_path,  "bestMeanModel.pt"))
            
            
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

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
            
            #print('final rewards',final_rewards.data)
            #('min reward', final_rewards.min())
            #print('median reward ', final_rewards.median() )
            
            infoToSave['timestep']+=[total_num_steps]
            infoToSave['FPS']+=[int(total_num_steps / (end - start))]
            infoToSave['meanReward']+=[final_rewards.mean()]
            infoToSave['medianReward']+=[final_rewards.median()]
            infoToSave['minReward']+=[final_rewards.min()]
            infoToSave['maxReward']+=[final_rewards.max()]
            infoToSave['entropy']+=[dist_entropy.data[0]]
            infoToSave['entropyCoef']+=[entropy_coef]

            infoToSave['valueLoss']+=[value_loss.data[0]]
            infoToSave['actionLoss']+=[action_loss.data[0]]
            
            infoToSave['numberOfChoices_Teacher']+=[currentCount['numberOfChoices_Teacher']]
            infoToSave['numberOfChoices_Agent']+=[currentCount['numberOfChoices_Agent']]
            
            updateRatioActions(infoToSave['actionRatio'],currentCount['numberOfChoices_Agent'],currentCount['numberOfChoices_Teacher'])
            
            
            infoToSave['envFinished']+=[np.sum(current_envFinished)]

            
            with open(os.path.join(save_path,'data.json'),'w') as fp:
                fp.write(json.dumps(infoToSave))

            
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                #if j>0:
                win = visdom_plot(viz, win, save_path, args.env_name, args.algo,infoToSave=infoToSave,actionDescription=actionDescription,numProcesses=args.num_processes)
            except IOError:
                pass
            
            if not args.earlySuccess is False:
                if numberOfSuccess>=args.earlySuccess:
                    print('end of the experiment')
                    print('the agent solved {} times the env'.format(numberOfSuccess))

                    f= open(fileInfo,"a")
                    
                    #save the number of timeSteps required
                    total_num_steps = (j + 1) * args.num_processes * args.num_steps
                    newLine='experience done in {} timeSteps \n'.format(total_num_steps)
                    print(newLine)
                    f.write(newLine)
                    
                    newLine='experience required {} interactions \n'.format(interactionCounter)
                    print(newLine)
                    f.write(newLine)
                    
                    
                    #save the date of end
                    newLine="experience finished on : {} at {}  \n".format(time.strftime("%d/%m/%Y"),time.strftime("%H:%M:%S"))
                    f.write(newLine)
                    
                    
                    f.close()
                    print(newLine)
                    #return(0)
                    useMissionAdvice+=1
                    print('raising difficulty, useMissionAdvice',useMissionAdvice)
                    entropyTime=0
        
        if not args.earlySuccess is False:
                #early stopping if the env has been cracked 
            if np.sum(current_envFinished) >=args.num_processes:
                numberOfSuccess+=1
            else:
                numberOfSuccess=0
    

if __name__ == "__main__":
    main()
