import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import glob

"""
File used to create a graph from a csv file and the name of the columns that need to be used
"""

def plotResult(columnNameX,columnNameY,columnNameZ,fileName,resultFileName):
    columnNameX = columnNameX
    columnNameY = columnNameY
    columnNameZ = columnNameZ
    fileName = fileName
    resultFileName = resultFileName

    x = []
    y = []
    z = []
    column1 = 0
    column2 = 0
    column3 = 0
    with open(fileName, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        firstLine = True
        for row in plots:
            if firstLine:
                for column in range(0, len(row)):
                    if row[column] == columnNameX:
                        column1 = column
                    if row[column] == columnNameY:
                        column2 = column
                    if row[column] == columnNameZ:
                        column3 = column
                firstLine = False
            else:
                x.append(int(row[column1]))
                y.append(float(row[column2]))
                z.append(float(row[column3]))
    figure = plt.figure()
    plt.plot(x, y,'r',label='N Step Before Done')
    plt.plot(x, z,'b',label='N Goal Reach')
    plt.legend()
    plt.xlabel('numberOfStep')
    figure.savefig(resultFileName)

def autoPlot(columnNameX,columnNameY,columnNameZ):
    for csvFile in glob.glob("evaluations/*.csv"):
        str = csvFile
        str = str.replace(".csv",".pdf")
        str = str.replace("evaluations/","results/")
        plotResult(columnNameX,columnNameY,columnNameZ,csvFile,str)



autoPlot("N_updates","N_step_before_done","N_goal_reached")