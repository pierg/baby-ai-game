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
    array = [[] for i in range(0,len(tab)*2+1)]
    column_number = [0 for i in range(0,len(tab)*2+1)]
    list_of_name = ["" for i in range(0,len(tab)*2+1)]
    list_of_name[0] = scale
    cpt = 1
    for x,y,z in tab:
        list_of_name[cpt] = x
        cpt += 1
        list_of_name[cpt] = y
        cpt += 1

    with open(fileName, 'r') as csvfile:
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
                for i in range(0,len(tab)*2+1):
                    array[i].append((float(row[column_number[i]])))

    pp = PdfPages(resultFileName)
    title = get_config_from_name(fileName)
    i = 0
    for x, y, z in tab:
        plt.figure()
        color = 'g'
        i += 1
        if len(array[i])>0:
            ymax = max(array[i])
            xpos = array[i].index(ymax)
            xmax = array[0][xpos]
            plt.plot(array[0], array[i],color,label=x)
            plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax,ymax-1 if ymax < 5 else ymax+5))
        i += 1
        if len(array[i])>0:
            color = 'b'
            ymax = max(array[i])
            xpos = array[i].index(ymax)
            xmax = array[0][xpos]
            plt.plot(array[0], array[i],color,label=y)
            plt.annotate(ymax, xy=(xmax, ymax), xytext=(xmax,ymax-1 if ymax < 5 else ymax+5))
        if z:
            area_top = []
            area_bot = []
            for j in range(0, len(array[i])):
                area_top.append(array[i-1][j] + array[i][j])
                area_bot.append(array[i-1][j] - array[i][j])
            plt.fill_between(array[0], area_bot, area_top, color="skyblue", alpha=0.4)
        plt.title(title)
        plt.xlabel('N Updates')
        plt.savefig(pp,format='pdf')

    plt.figure()
    img = get_image_from_name("configurations/*",fileName)
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
    for csvFile in glob.glob("evaluations/*.csv"):
        name = csvFile
        random_number = randint(0,999999)
        name = name.replace(".csv",str(random_number))
        name += str(".pdf")
        name = name.replace("evaluations/","results/")
        plot_result(scale,tab,csvFile,name)


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