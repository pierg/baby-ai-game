# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})

plotStatsDoor=False

def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(31, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size):
    datas = []
    infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')    


                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fix_point(x, y, bin_size)
    return [x, y]


color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]



def plotActionRatioWithVisdom(timestep,actionRatio,name,game,viz,win,folder,actionDescription=None):
    
# =============================================================================
#     Plot the action ratio
# =============================================================================
    #get the number of different actions possible
    n_groups = len(actionDescription)

    #get the names of the action ID
    names=[0 for i in range(n_groups)]
    for key,value in actionDescription.items():
        names[value]=key 
    
    
    fig, ax = plt.subplots()
    
    localWin =viz.line(
       Y=np.column_stack([actionRatio[indexAction] for indexAction in range(n_groups)]),
       X=np.column_stack([timestep for indexAction in range(n_groups)]),
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='Timestep',
            ylabel='ration of the actions, NactionAgent/NactionTeacher',
            title='evolution of the ratios of selected actions',
            legend=[names[indexAction] for indexAction in range(n_groups)],
        ),win=win['actionRatio']
)
    
    
    for indexAction in range(n_groups):
        plt.plot(timestep,actionRatio[indexAction],label=names[indexAction])
        
        
    
    plt.xlabel('Number of timestep')
    plt.ylabel('ration Choice Agent/Choice Teacher')
    plt.title('ratio of action choices -- -1 meaning Teacher never suggested this action')
    ax.set_ylim(-1.5,10)

    plt.legend()
    
    fig.savefig(os.path.join(folder,'actionRatio.png'))


    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    #image = np.transpose(image, (2, 0, 1))
    win['actionRatio']=localWin
    
    
    

    return(win)
    
def plotEntropyCoefWithVisdom(timestep,entropyCoef,name,game,viz,win,folder,actionDescription=None):
    
    
    
    fig, ax = plt.subplots()
    
    
    plt.plot(timestep,entropyCoef)
    
    localWin =viz.line(
       Y=np.array(entropyCoef),
       X=np.array(timestep),
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='Timestep',
            ylabel='entropy coefficient',
            title='evolution of the enctropy coefficient',
            legend=['entropyCoef'],
        ),win=win['entropyCoef']
)
        
        
    
    plt.xlabel('Number of timestep')
    plt.ylabel('entropy coefficient')
    plt.title('evolution of the entropy coefficient')

    plt.legend()
    
    fig.savefig(os.path.join(folder,'entropyCoef.png'))


    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    win['entropyCoef']=localWin
    
    

    return(win)

def plotEnvFinishedWithVisdom(timestep,envFinished,numProcesses,name,game,viz,win,folder,actionDescription=None):
    
    
    
    fig, ax = plt.subplots()
    
    y1=[numProcesses for i in range(len(timestep))]
    
    localWin =viz.line(
       Y=np.column_stack([y1, envFinished]),
       X=np.column_stack([timestep, timestep]),
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='Timestep',
            ylabel='number of envs that won the game',
            title='evolution of the success of the envs',
            legend=['total number of agents', 'winning agents'],
        ),win=win['envFinished']
)
    
    
    
    plt.plot(timestep,[numProcesses for i in range(len(timestep))],label='total number of envs')
    plt.plot(timestep,envFinished,label='number of envs won')
        
        
    
    plt.xlabel('Number of timestep')
    plt.ylabel('number of agents that won the game')
    plt.title('evolution of the performance of the agents')

    plt.legend()
    
    fig.savefig(os.path.join(folder,'envFinished.png'))


    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    win['envFinished']=localWin
    
    

    return(win)

def plotStatsDoorWithVisdom(timestep,doorMet,doorOpened,maxDoorOpened,maxDoorMet,name,game,viz,win,folder,actionDescription=None):
    
    
    
    
# =============================================================================
#     Plot the total number of doors met
# =============================================================================
    fig, ax = plt.subplots()
    
    plt.plot(timestep,doorMet,label='number of doors met')

    #X=np.array(doorMet)
    #print('shape', X.shape)
    localWin =viz.bar(
       X=np.column_stack(doorMet),
        opts=dict(
            rownames=[str(i) for i in timestep],

            width=600,
            height=600,
            xlabel='Timestep',
            ylabel='number of doors met in the game',
            title='evolution of the number of doors met',
        ),win=win['doorMet']
)
           
        
      
        
    
    plt.xlabel('Number of timestep')
    plt.ylabel('door stats')
    plt.title('doors met ')

    plt.legend()
    
    fig.savefig(os.path.join(folder,'doorMet.png'))


    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    win['doorMet']=localWin 
    
    
    
    
# =============================================================================
#     Plot the total number of doors opened
# =============================================================================
    fig, ax = plt.subplots()
    
    plt.plot(timestep,doorOpened,label='number of doors opened')
    
    
    localWin =viz.bar(
   X=np.column_stack(doorOpened),
    opts=dict(
        rownames=[str(i) for i in timestep],

        width=600,
        height=600,
        xlabel='Timestep',
        ylabel='number of doors opened in the game',
        title='evolution of the number of doors opened',
    ),win=win['doorOpened']
)    
        
    
    plt.xlabel('Number of timestep')
    plt.ylabel('door stats')
    plt.title('doors opened')

    plt.legend()
    
    fig.savefig(os.path.join(folder,'doorOpened.png'))


    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    win['doorOpened']=localWin
    
    
    
    
# =============================================================================
#  Plot the same stats for the best agent
#    
#    Number of doors opened
# =============================================================================
    
    fig, ax = plt.subplots()
    
    plt.plot(timestep,maxDoorOpened,label='number of doors opened')
    localWin =viz.bar(
   X=np.column_stack(maxDoorOpened),
    opts=dict(
        rownames=[str(i) for i in timestep],

        width=600,
        height=600,
        xlabel='Timestep',
        ylabel=' number of doors opened in the game for the best env',
        title='evolution of the success of the best env --- doors opened',
        color='red'
    ),win=win['maxDoorOpened']
)    
            
        
    
    plt.xlabel('Number of timestep')
    plt.ylabel('door stats')
    plt.title('evolution of the best success')

    plt.legend()
    
    fig.savefig(os.path.join(folder,'maxDoorOpened.png'))


    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    win['maxDoorOpened']=localWin
    
    
    
# =============================================================================
#     Number of doors met for the bet env
# =============================================================================
    fig, ax = plt.subplots()
    
    plt.plot(timestep,maxDoorMet,label='number of doors met')
    localWin =viz.bar(
   X=np.column_stack(maxDoorMet),
    opts=dict(
        rownames=[str(i) for i in timestep],

        width=600,
        height=600,
        xlabel='Timestep',
        ylabel=' number of doors met in the game for the best env',
        title='evolution of the success of the best env --- doors met',
        color='red'

    ),win=win['maxDoorMet']
)        
        
    
    plt.xlabel('Number of timestep')
    plt.ylabel('door stats')
    plt.title('evolution of the best success')

    plt.legend()
    
    fig.savefig(os.path.join(folder,'maxDoorMet.png'))


    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    win['maxDoorMet']=localWin 
    
    
    

    return(win)
    

def plotStatsActionsWithVisdom(timestep,numberOfChoices_Agent,numberOfChoices_Teacher,name,game,viz,win,folder,actionDescription=None):
    
# =============================================================================
#     Plot the action statistics
# =============================================================================
    #get the number of different actions possible
    n_groups = len(actionDescription)

    #get the names of the action ID
    names=[0 for i in range(n_groups)]
    for key,value in actionDescription.items():
        names[value]=key 
    
   
    
    fig, ax = plt.subplots()
    
    index = np.arange(n_groups)
    bar_width = 0.35
    
    opacity = 0.4
    error_config = {'ecolor': '0.3'}  
    
    rects1 = ax.bar(index, numberOfChoices_Agent[-1], bar_width,
                    alpha=opacity, color='b',
                    error_kw=error_config,
                    label='Agent Choices')
    
    rects2 = ax.bar(index + bar_width, numberOfChoices_Teacher[-1], bar_width,
                    alpha=opacity, color='r',
                     error_kw=error_config,
                    label='Teacher Choices')
    
    ax.set_xlabel('Action ID')
    ax.set_ylabel('Number of realisations - since the beginning of the game')
    ax.set_title('Action Statistics')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(names)
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(folder,'statsAction.png'))


    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    win['statsAction']=viz.image(image,win['statsAction'])
    
    
    

    return(win)
    
    

def plotEntropyWithVisdom(tx,ty,name,game,viz,win,folder,ylabel='entropy',xlabel='Number of Timesteps'):
    fig = plt.figure()
    plt.plot(tx, ty, label="entropy of the action distribution")
    localWin =viz.line(
       Y=np.array(ty),
       X=np.array(tx),
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='Timestep',
            ylabel='entropy',
            title='evolution of the entropy',
            legend=['emtropy'],
        ),win=win['entropy']
)


    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


    plt.title(game)
    plt.legend(loc=4)
    fig.savefig(os.path.join(folder,'entropy.png'))

    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    win['entropy']=localWin
    return(win)
    
    
def plotRewardsWithVisdom(timestep,meanReward,medianReward,minReward,maxReward,
                          name,game,viz,win,folder,ylabel='Rewards',xlabel='Number of Timesteps'):
    fig = plt.figure()
    
    plt.plot(timestep,minReward,label='min reward',color='red')
    plt.plot(timestep,maxReward,label='max reward',color='blue')
    #print('timesteps',timestep)
    #print('min', minReward)
    #print('max',maxReward)
    #print('median',medianReward)
    plt.fill_between(timestep, minReward, maxReward, color='blue', alpha='0.2')

    plt.plot(timestep,meanReward,label='mean reward',color='green')
    plt.plot(timestep,medianReward,label='median reward',color='yellow')
    
    
    localWin =viz.line(
       Y=np.column_stack([minReward,maxReward,meanReward,medianReward]),
       X=np.column_stack([timestep for i in range(4)]),
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='Timestep',
            ylabel='reward',
            title='evolution of the rewards',
            legend=['minReward', 'maxReward', 'meanReward', 'medianReward'],
        ),win=win['rewards']
)





    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


    plt.title(game)
    plt.legend(loc=4)
    fig.savefig(os.path.join(folder,'rewards.png'))
    plt.show()
    plt.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    plt.close(fig)

    # Show it in visdom
    image = np.transpose(image, (2, 0, 1))
    win['rewards']=localWin
    return(win)
    
    
exclude=['updates','timestep','numberOfChoices_Teacher','numberOfChoices_Agent']
def visdom_plot(viz, win, folder, game, name, numProcesses,bin_size=100, smooth=1,infoToSave=None,actionDescription=None):
    global plotStatsDoor
    
    timestep=infoToSave['timestep']
    meanReward=infoToSave['meanReward']
    medianReward=infoToSave['medianReward']
    minReward=infoToSave['minReward']
    maxReward=infoToSave['maxReward']
    win=plotRewardsWithVisdom(timestep,meanReward,medianReward,minReward,maxReward,name,game,viz,win,folder)    
    
       
    entropy=infoToSave['entropy']
    win=plotEntropyWithVisdom(timestep,entropy,name,game,viz,win,folder)
    
    
    numberOfChoices_Teacher=infoToSave['numberOfChoices_Teacher']
    numberOfChoices_Agent=infoToSave['numberOfChoices_Agent']
    win=plotStatsActionsWithVisdom(timestep,numberOfChoices_Agent,numberOfChoices_Teacher,name,game,viz,win,folder,actionDescription=actionDescription)
    
    
    actionRatio=infoToSave['actionRatio']    
    win=plotActionRatioWithVisdom(timestep,actionRatio,name,game,viz,win,folder,actionDescription=actionDescription)


    entropyCoef=infoToSave['entropyCoef']
    win=plotEntropyCoefWithVisdom(timestep,entropyCoef,name,game,viz,win,folder,actionDescription=actionDescription)


    envFinished=infoToSave['envFinished']
    win=plotEnvFinishedWithVisdom(timestep,envFinished,numProcesses,name,game,viz,win,folder,actionDescription=actionDescription)

    doorMet=infoToSave['doorMet']
    doorOpened=infoToSave['doorOpened']
    maxDoorOpened=infoToSave['maxDoorOpened']
    maxDoorMet=infoToSave['maxDoorMet']

    if plotStatsDoor:
        win=plotStatsDoorWithVisdom(timestep,doorMet,doorOpened,maxDoorOpened,maxDoorMet,name,game,viz,win,folder,actionDescription=actionDescription)
    else:
        plotStatsDoor=True
        
        
    return (win)


if __name__ == "__main__":
    from visdom import Visdom
    viz = Visdom(server='http://eos11',port=24431)
    visdom_plot(viz, None, '/tmp/gym/', 'BreakOut', 'a2c', bin_size=100, smooth=1)

