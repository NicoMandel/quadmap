import quadtree as qt
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plotdir(dirtoplot):
    file_list = glob.glob(dirtoplot + "/*.pkl")
    no_exp = len(file_list)
    rs = np.ceil(np.sqrt(no_exp)).astype(np.int)
    cls = np.ceil(no_exp / rs).astype(np.int)
    fig, axs = plt.subplots(rs, cls)
    for i, f in enumerate(file_list):
        exp = resolvename(f)
        c = i % cls
        r = i // cls
        t = qt.Quadtree.load(f)
        print("loading tree {} done. proceeding with pruning".format(exp))
        t.postprocess()
        print("pruning tree {} done. Proceeding with plotting".format(exp))
        t.plot_tree(axs[r,c])
        axs[r,c].set_title(exp)
    
    plt.show()

def findexp(exp_name, directory):
    # file_list = glob.glob(directory + "*{}*".format(exp_name))
    file_list = [f for f in os.listdir(directory) if exp_name in f and ".pkl" in f]
    return file_list[0]


def resolvename(f):
    fname = f.split("/")[-1]
    date_exp = fname.split("-")[3:5]
    exp = "_".join(date_exp)
    return exp

def plotexperiment(directory, experiment_name, plot_title, depth=10):
    fname = os.path.join(directory,experiment_name)
    tree = qt.Quadtree.load(fname)
    print("Loading of {} done. Proceeding with pruning".format(experiment_name))
    tree.postprocess(depth=depth)

    print("Pruning done, Proceeding with Plotting")

    fig = plt.figure(figsize=(15,9))
    ax = fig.gca()
    tree.plot_tree(ax, depth=depth)
    ax.set_title("{}: {}".format(plot_title, depth))

    plt.show()

if __name__=="__main__":
    thisdir = os.path.dirname(__file__)
    
    a = datetime(2021, 10, 6, 18, 0)
    outdir_time = a.strftime("%y-%m-%d_%H-%M")
    # b = datetime(2021, 10, 7, 5, 32).strftime("%y-%m-%d_%H-%M")
    c = datetime(2021, 10, 7)
    cform = c.strftime("%y-%m-%d")
    exx = "sim"
    mode="hyb"
    setting = "20"
    experiment = "{}_{}-{}".format(exx, mode, setting)
    
    max_depth = 16
    low = (-90,-30)
    scale = 250
    
    outputdir = os.path.abspath(os.path.join(thisdir, '..', 'output', cform))

    # f = "{}_{}-qt-{}-{}-{}.pkl".format(b, experiment, max_depth, low, scale)
    f = findexp(experiment, outputdir)
    plotexperiment(outputdir, f, experiment, depth=10)
    # c = datetime(2021, 10, 7)
    # cform = c.strftime("%y-%m-%d")
    # dirtoplot = os.path.abspath(os.path.join(thisdir, '..', 'output', cform))
    # plotdir(dirtoplot)
    
    print("Plotting done")