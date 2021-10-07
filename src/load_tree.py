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

        

def resolvename(f):
    fname = f.split("/")[-1]
    date_exp = fname.split("-")[3:5]
    exp = "_".join(date_exp)
    return exp

def plotexperiment(directory, experiment_name, depth=10):
    fname = os.path.join(directory,experiment_name)
    tree = qt.Quadtree.load(fname)
    n = resolvename(experiment_name)
    print("Loading done. Proceeding with pruning")
    tree.postprocess(depth=depth)

    print("Pruning done, Proceeding with Plotting")

    fig = plt.figure(figsize=(15,9))
    ax = fig.gca()
    tree.plot_tree(ax)
    ax.set_title("{}: {}".format(n, depth))

    plt.show()

if __name__=="__main__":
    thisdir = os.path.dirname(__file__)
    
    a = datetime(2021, 10, 6, 18, 0)
    outdir_time = a.strftime("%y-%m-%d_%H-%M")
    b = datetime(2021, 10, 6, 21, 9).strftime("%y-%m-%d_%H-%M")
    experiment = "exp_tgt1-descend"
    max_depth = 16
    low = (-90,-30)
    scale = 250
    
    outputdir = os.path.abspath(os.path.join(thisdir, '..', 'output', 'sim', outdir_time))

    f = "{}_{}-qt-{}-{}-{}.pkl".format(b, experiment, max_depth, low, scale)
    plotexperiment(outputdir, f)
    # plotdir(outputdir)
    
    print("Plotting done")