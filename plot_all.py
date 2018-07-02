import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import glob
from random import randint
import configurations.config_grabber as cg
from pathlib import Path
import screenHelper
"""
File used to create a graph from a csv file and the name of the columns that need to be used
"""


def plot_result(scale,tab,fileName,resultFileName):
    pp = PdfPages(resultFileName)
    title = get_config_from_name(fileName[0])

    array = [[[] for i in range(0, len(tab) * 2 + 1)] for e in range(len(fileName))]
    mean_array = [[] for i in range(0,len(tab)*2+1)]
    column_number =[0 for i in range(0,len(tab)*2+1)]
    list_of_name = ["" for i in range(0, len(tab)*2+1)]
    list_of_name[0] = scale
    cpt = 1
    last_mean = [float(0) for i in range ( 0, 22)]
    one_process_max = 0
    all_process_max = 0

    for x,y,z in tab:
        list_of_name[cpt] = x
        cpt += 1
        list_of_name[cpt] = y
        cpt += 1
    for t in range(0, len(fileName)):
        with open(fileName[t], 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            first_line = True
            for row in plots:
                if first_line:
                    for column in range(0, len(row)):
                        for i in range(0, len(list_of_name)):
                            if list_of_name[i] == row[column]:
                                column_number[i] = column
                    first_line = False
                else:
                    for i in range(0,len(tab)*2 + 1):
                        array[t][i].append((float(row[column_number[i]])))
            for k in range (len(last_mean)):
                last_mean[k] += float(row[k])
            for i in range(0, len(tab) * 2 + 1):
                for j in range(0,len(array[t][i])):
                    if j < len(mean_array[i]) and len(mean_array[i]) != 0:
                        mean_array[i][j][0] = (mean_array[i][j][0]*mean_array[i][j][1] + array[t][i][j])/(mean_array[i][j][1] + 1)
                        mean_array[i][j][1] += 1
                    else:
                        mean_array[i].append([array[t][i][j],1])

    for t in range (0, len(array)):
        pmax = max(array[t][-2])
        pos = array[t][-2].index(pmax)
        all_process_max += array[t][0][pos]
    all_process_max = all_process_max/len(array)
    print ("all_process_max : ",all_process_max)
    for k in range (len(last_mean)):
        last_mean[k] = last_mean[k] / len(fileName)
    print(last_mean)

    for t in range(0, len(mean_array[0])):
        for j in range(len(mean_array)):
            mean_array[j][t] = mean_array[j][t][0]


    plt.figure()
    for t in range(0, len(fileName)):
        if len(array[t][1]) > 0:
            color = 0
            ymax = max(array[t][1])
            xpos = array[t][1].index(ymax)
            xmax = array[t][0][xpos]
            ymax = float("{0:.1f}".format(ymax))
            plt.plot(array[t][0], array[t][1], color + t * 5)
            plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax))
    plt.ylabel(str(list_of_name[1]))
    plt.title(title)
    plt.xlabel('N Updates')
    plt.savefig(pp, format='pdf')

    plt.figure()
    for t in range(0, len(fileName)):
        if len(array[t][2]) > 0:
            color = 0
            ymax = max(array[t][2])
            xpos = array[t][2].index(ymax)
            xmax = array[t][0][xpos]
            ymax = float("{0:.1f}".format(ymax))
            plt.plot(array[t][0], array[t][2], color + t * 5)
            plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax))

    plt.ylabel(str(list_of_name[2]))
    plt.title(title)
    plt.xlabel('N Updates')
    plt.savefig(pp, format='pdf')


    plt.figure()
    if len(mean_array[2]) > 0:
        ymax = max(mean_array[2])
        xpos = mean_array[2].index(ymax)
        xmax = mean_array[0][xpos]
        ymax = float("{0:.1f}".format(ymax))
        plt.plot(mean_array[0], mean_array[2], 'y', linewidth=2.5, label=list_of_name[2] + "_mean")
        plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax))

    if len(mean_array[1]) > 0:
        ymax = max(mean_array[1])
        xpos = mean_array[1].index(ymax)
        xmax = mean_array[0][xpos]
        ymax = float("{0:.1f}".format(ymax))
        plt.plot(mean_array[0], mean_array[1], 'r', linewidth=2.5, label=list_of_name[1] + "_mean")
        plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax))

    plt.legend()
    plt.title(title)
    plt.xlabel('N Updates')
    plt.savefig(pp, format='pdf')



    i = 1
    for x, y, z in tab:
        if x == list_of_name[1]:
            i += 2
            continue
        plt.figure()
        for t in range(0, len(fileName)):
            color = 'g'
            if len(array[t][i])>0:
                ymax = max(array[t][i])
                xpos = array[t][i].index(ymax)
                xmax = array[t][0][xpos]
                ymax = float("{0:.1f}".format(ymax))
                if t == 0 :
                    plt.plot(array[t][0], array[t][i],color,label=x)
                else:
                    plt.plot(array[t][0], array[t][i], color)
                plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax,ymax))
            if len(array[t][i + 1])>0:
                color = 'b'
                ymax = max(array[t][i + 1])
                xpos = array[t][i + 1].index(ymax)
                xmax = array[t][0][xpos]
                ymax = float("{0:.1f}".format(ymax))
                if t == 0:
                    plt.plot(array[t][0], array[t][i + 1],color,label=y)
                else:
                    plt.plot(array[t][0], array[t][i + 1], color)
                plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax,ymax))

            if z:
                area_top = []
                area_bot = []
                for k in range(0, len(array[t][i + 1])):
                    area_top.append(array[t][i][k] + array[t][i + 1][k])
                    area_bot.append(array[t][i][k] - array[t][i + 1][k])
                plt.fill_between(array[t][0], area_bot, area_top, color="skyblue", alpha=0.4)

        color = 'r'
        if len(mean_array[i]) > 0:
            ymax = max(mean_array[i])
            xpos = mean_array[i].index(ymax)
            xmax = mean_array[0][xpos]
            ymax = float("{0:.1f}".format(ymax))
            plt.plot(mean_array[0], mean_array[i], color, linewidth = 2.5, label=x + "_mean")
            plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax))
        if len(mean_array[i + 1]) > 0:
            color = 'y'
            ymax = max(mean_array[i + 1])
            xpos = mean_array[i + 1].index(ymax)
            xmax = mean_array[0][xpos]
            ymax = float("{0:.1f}".format(ymax))
            plt.plot(mean_array[0], mean_array[i + 1], color, linewidth = 2.5,label=y + "_mean")
            plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax, ymax))

        if z:
            area_top = []
            area_bot = []
            for k in range(0, len(mean_array[i + 1])):
                area_top.append(mean_array[i][k] + mean_array[i + 1][k])
                area_bot.append(mean_array[i][k] - mean_array[i + 1][k])
            plt.fill_between(mean_array[0], area_bot, area_top, color="skyblue", alpha=0.4)


        i += 2

        plt.legend()
        plt.title(title)
        plt.xlabel('N Updates')
        plt.savefig(pp,format='pdf')

    plt.figure()
    img = get_image_from_name("configurations/*",fileName[0])
    if img is not None:
        if "main" in img:
            screenHelper.main()
        img = plt.imread(img)
        plt.imshow(img)
        plt.savefig(pp, format='pdf')
    pp.close()
    print(resultFileName, " generated")


def get_config_from_name(file):
    try :
        file = file.split(".csv")[0]
        file = file.split("evaluations/")[1]
        config = cg.Configuration.grab(file)
    except FileNotFoundError:
        config = cg.Configuration.grab()
    title = "Monitors : "
    for typeOfMonitor in config.monitors:
        for monitors in typeOfMonitor:
            for monitor in monitors:
                if monitor.active:
                    title += monitor.type + "_" + monitor.name
    rewards = "reward goal : {0} ".format(config.rewards.standard.goal)
    rewards += "/ step : {0} ".format(config.rewards.standard.step)
    rewards += "/ death : {0} ".format(config.rewards.standard.death)
    return title + "\n" + rewards



def get_image_from_name(path,my_file):
    for file in glob.glob(path):
        if ".json" in file:
            file = file.split("configurations/")[1]
            file = file.split(".json")[0]
            try:
                config = cg.Configuration.grab(file)
            except FileNotFoundError:
                config = cg.Configuration.grab()
            my_file = my_file.replace("evaluations/","")
            my_file = my_file.replace(".csv", "")
            if config.config_name == my_file:
                if "randoms" in path:
                    return "results/screens/randoms/"+config.env_name+".png"
                if "crafted" in path:
                    return "results/screens/crafted/"+config.env_name+".png"
                return "results/screens/"+config.env_name+".png"
    if "crafted" not in path and "randoms" not in path:
        return get_image_from_name("configurations/crafted/*",my_file)
    elif "randoms" not in path:
        return get_image_from_name("configurations/randoms/*",my_file)
    else:
        return None

def autoPlot(scale,tab):
    for csvFile in glob.glob("evaluations/*_0.csv"):
        csvFileQueue = []
        csvFileQueue.append(csvFile)
        i = 1
        name = csvFile
        random_number = randint(0,999999)
        name = name.replace("_0.csv","")
        while True:
            if (glob.glob(name +"_%s.csv" % i)):
                for csvNext in (glob.glob(name +"_%s.csv" % i)):
                    csvFileQueue.append(csvNext)
                i += 1
                continue
            break
        name = name + (str(random_number))
        name += str(".pdf")
        name = name.replace("evaluations/","results/")
        plot_result(scale,tab,csvFileQueue,name)


def create_all_images(path):
    for file in glob.glob(path):
        # Folder except rewards and environments
        if "." not in file and "rewards" not in file and "environments" not in file:
            create_all_images(file+"/*")
        elif ".json" in file:
            # image need to be created
            file = file.split("configurations/")[1]
            file = file.split(".json")[0]
            if not check_if_image_already_exist(file):
                screenHelper.main(file)


def check_if_image_already_exist(path):
    print(path)
    config = cg.Configuration.grab(path)
    folder = ""
    if "crafted" in path:
        folder = "crafted/"
    elif "randoms" in path:
        folder = "randoms/"
    path = path.replace(path,"results/screens/"+folder+config.env_name+".png")
    path = Path(path)
    if path.exists():
        return True
    return False

create_all_images("configurations/*")
print("all images created")

scale = "N_updates"
first_graph = ("N_step_AVG","N_goal_reached",False)
second_graph = ("N_death","N_saved",False)
third_graph = ("Reward_mean","Reward_std",True)
tab = (first_graph,second_graph,third_graph)
autoPlot(scale, tab)