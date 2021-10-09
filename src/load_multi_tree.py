import quadtree as qt
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
import re

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
    return file_list

def resolvename(f):
    fname = f.split("/")[-1]
    date_exp = fname.split("-")[3:5]
    exp = "_".join(date_exp)
    return exp

def plotTwoTrees(directory, experiment_name, title="", depth=4):
    fname1 = os.path.join(directory,experiment_name[0])
    fname2 = os.path.join(directory,experiment_name[1])
    tree_1 = qt.Quadtree.load(fname1)
    tree_2 = qt.Quadtree.load(fname2)
    tit_1 = regexFilename(experiment_name[0])
    tit_2 = regexFilename(experiment_name[1])
    print("Loading of {} done. Proceeding with pruning".format(experiment_name))
    tree_1.postprocess(depth=depth)
    tree_2.postprocess(depth=depth)

    print("Pruning done, Proceeding with Plotting")
    fig, axs = plt.subplots(1, 2)
    tree_1.plot_tree(axs[0], depth=depth)
    axs[0].set_title("{}: {}".format("Tree {}".format(tit_1), depth))
    tree_2.plot_tree(axs[1], depth=depth)
    axs[1].set_title("{}: {}".format("Tree {}".format(tit_2), depth))
    plt.suptitle(title)
    plt.show()

def regexFilename(name):
    out = re.search(r'qt(\d?)',name)
    return out.group()

def kl_div(vec_true, vec_pred):
    """
    use the definition of the KL divergence
        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        * look at octomap paper for their defintion - the same!
    """
    return np.sum(vec_true*np.log(vec_pred)) * (-1.0)

def parse_args(defaultdir):
    """
        Function to parse the arguments 
    """
    parser = ArgumentParser(description="Plotting two quadtrees that have been generated by the same file.")
    parser.add_argument("-d", "--depth", help="Plotting depth of the quadtree. Default is 1.", type=int,default=10)
    parser.add_argument("--input", help="Input directory. Default is the one of the file", type=str, default=defaultdir)
    parser.add_argument("--file", help="Input Experiment. Used to read the file which contains the string given by file from input directory. If not specified will raise FileNotFoundError", type=str, default="exp_tgt2-descend")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    thisdir = os.path.dirname(__file__)
    a = datetime(2021, 10, 8)
    outdir_date = a.strftime("%y-%m-%d")
    defdir = os.path.join(thisdir, '..', 'output', 'hpc', outdir_date)
    args = parse_args(defdir)
    
    f = findexp(args.file, args.input)
    plotTwoTrees(args.input, f, args.file, depth=args.depth)
    # cform = c.strftime("%y-%m-%d")
    # dirtoplot = os.path.abspath(os.path.join(thisdir, '..', 'output', cform))
    # plotdir(dirtoplot)
    
    print("Plotting done")