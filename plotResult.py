import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import glob
from random import randint
"""
File used to create a graph from a csv file and the name of the columns that need to be used
"""

def plotResult(tab,fileName,resultFileName):
    fileName = fileName
    resultFileName = resultFileName
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

    ymax = max(array[1])
    xpos = array[1].index(ymax)
    xmax = array[0][xpos]

    pp = PdfPages(resultFileName)
    plt.figure()
    plt.plot(array[0], array[1],'r',label='N Step AVG')
    plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax+5))

    ymax = max(array[2])
    xpos = array[2].index(ymax)
    xmax = array[0][xpos]

    plt.plot(array[0], array[2],'b',label='N Goal Reached')
    plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax+5))

    plt.legend()
    plt.xlabel('N Updates')
    plt.savefig(pp,format='pdf')
    plt.figure()

    ymax = max(array[9])
    xpos = array[9].index(ymax)
    xmax = array[0][xpos]

    plt.plot(array[0], array[9],'r',label='N break')
    plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax+5))

    ymax = max(array[4])
    xpos = array[4].index(ymax)
    xmax = array[0][xpos]

    plt.plot(array[0], array[4],'b',label='N Saved')
    plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax+5))

    ymax = max(array[5])
    xpos = array[5].index(ymax)
    xmax = array[0][xpos]

    plt.plot(array[0], array[5],'g',label='N Episode')
    plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax+5))


    plt.legend()
    plt.xlabel('N Updates')
    plt.savefig(pp, format='pdf')
    plt.figure()

    ymax = max(array[6])
    xpos = array[6].index(ymax)
    xmax = array[0][xpos]

    plt.plot(array[0], array[6], 'b', label='Reward mean')
    plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax + 5))

    ymax = max(array[7])
    xpos = array[7].index(ymax)
    xmax = array[0][xpos]

    plt.plot(array[0], array[7], 'r', label='Reward max')
    plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax + 5))

    ymax = max(array[7])
    xpos = array[7].index(ymax)
    xmax = array[0][xpos]

    plt.plot(array[0], array[8], 'g', label='Reward min')
    plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax + 5))

    plt.legend()
    plt.xlabel('N Updates')
    plt.savefig(pp, format='pdf')
    plt.figure()
    pp.close()



def autoPlot(tab):
    for csvFile in glob.glob("evaluations/*.csv"):
        name = csvFile
        randomNumber = randint(0,999999)
        name = name.replace(".csv",str(randomNumber))
        name += str(".pdf")
        print(name)
        name = name.replace("evaluations/","results/")
        plotResult(tab,csvFile,name)


tab = ("N_updates","N_step_AVG","N_goal_reached","N_death","N_saved","N_Total_episodes","Reward_mean","Reward_max","Reward_min","N_break")
autoPlot(tab)