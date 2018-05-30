import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import glob
from random import randint
import configurations.config_grabber as cg

"""
File used to create a graph from a csv file and the name of the columns that need to be used
"""

def plotResult(tab,fileName,resultFileName):
    array = [[] for i in range(0,len(tab))]
    columnNumber = [0 for i in range(0,len(tab))]

    with open(fileName, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        firstLine = True
        for row in plots:
            if firstLine:
                for column in range(0, len(row)):
                    for i in range(0,len(tab)):
                        if tab[i] == row[column]:
                            columnNumber[i] = column
                firstLine = False
            else:
                for i in range(0,len(tab)):
                    array[i].append((float(row[columnNumber[i]])))

    pp = PdfPages(resultFileName)
    title = get_config_from_name(fileName)
    plt.figure()
    color = 'g'
    for i in range(1,len(tab)):
        ymax = max(array[i])
        xpos = array[i].index(ymax)
        xmax = array[0][xpos]
        plt.plot(array[0], array[i],color,label=tab[i])
        plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax,ymax-1 if ymax < 5 else ymax+5))
        if i%2 == 0:
            color = 'g'
            plt.legend()
            plt.title(title)
            plt.xlabel('N Updates')
            plt.savefig(pp,format='pdf')
            plt.figure()
        else:
            color = 'b'
    pp.close()

def get_config_from_name(file):
    file = file.split(".csv")[0]
    file = file.split("evaluations/")[1]
    config = cg.Configuration.grab(file)
    monitors = "Monitors : "
    if hasattr(config.monitors, 'absence'):
        for avoid_obj in config.monitors.absence.monitored:
            if avoid_obj.active:
                monitors += avoid_obj.name + " "
    if hasattr(config.monitors, 'precedence'):
        for precedence_obj in config.monitors.precedence.monitored:
            if precedence_obj.active:
                monitors += config.monitors + " "
    rewards = "reward goal : {0} ".format(config.reward.goal)
    rewards += "/ step : {0} ".format(config.reward.step)
    rewards += "/ death : {0} ".format(config.reward.death)
    return monitors + "\n" + rewards

def autoPlot(tab):
    for csvFile in glob.glob("evaluations/*.csv"):
        name = csvFile
        randomNumber = randint(0,999999)
        name = name.replace(".csv",str(randomNumber))
        name += str(".pdf")
        name = name.replace("evaluations/","results/")
        plotResult(tab,csvFile,name)


tab = ("N_updates","N_step_AVG","N_goal_reached","N_death","N_saved","Reward_mean","Reward_max")
autoPlot(tab)