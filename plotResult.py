import matplotlib.pyplot as plt
import csv

"""
File used to create a graph from a csv file and the name of the columns that need to be used
"""


# If you want to change the file or the columns used to create the graph do it here
columnNameX = "N_updates"
columnNameY = "Reward_mean"
fileName = "evaluations/water_deadend_32x32.csv"


x = []
y = []
column1 = 0
column2 = 0

with open(fileName,'r') as csvfile:
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

plt.plot(x,y)
plt.xlabel('numberOfStep')
plt.ylabel('reward_mean')
plt.show()