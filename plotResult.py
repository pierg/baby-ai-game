import matplotlib.pyplot as plt
import csv
import glob

"""
File used to create a graph from a csv file and the name of the columns that need to be used
"""

def plotResult(columnNameX,columnNameY,fileName,resultFileName):
    columnNameX = columnNameX
    columnNameY = columnNameY
    fileName = fileName
    resultFileName = resultFileName

    x = []
    y = []
    column1 = 0
    column2 = 0

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
                firstLine = False
            else:
                x.append(int(row[column1]))
                y.append(float(row[column2]))
    figure = plt.figure()
    plt.plot(x, y)
    plt.xlabel('numberOfStep')
    plt.ylabel('reward_mean')
    figure.savefig(resultFileName)

def autoPlot(columnNameX,columnNameY):
    for csvFile in glob.glob("evaluations/*.csv"):
        str = csvFile
        str = str.replace(".csv",".pdf")
        str = str.replace("evaluations/","results/")
        plotResult(columnNameX,columnNameY,csvFile,str)



autoPlot("N_updates","Reward_mean")