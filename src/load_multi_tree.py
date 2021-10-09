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

def compareexp(directory, exp_list, title="", depth=10):
    exp_ct = len(exp_list)
    tree_dict = {}
    for exp_fn in exp_list:
        fname = os.path.join(directory, exp_fn)
        tree = qt.Quadtree.load(fname)
        # tree.postprocess(depth=depth)
        exx = regexFilename(fname)
        skip = int(exx.split("-")[-1])
        fr = getFreq(skip, title)
        tree_dict[fr] = tree
    print("Loading trees and pruning for {} done, continuing with KL comparison and plotting".format(
        title
    ))
    base_freq = getBaseFreq(title)
    base_tree = tree_dict[base_freq]
    del tree_dict[base_freq]
    kl_dict = {}
    for k, v in tree_dict.items():
        kl_dict[k] = compareTwoTrees(base_tree, tree_comp=v, idx=k, d=depth)
    
    return kl_dict

def plot_kl_dict(kl_dict, ax, title, depth):
    
    # ndict = sorted(kl_dict)
    # x = np.asarray(list(ndict.keys()))
    # y = np.asarray(list(ndict.values()))
    x = list(kl_dict.keys())
    y = list(kl_dict.values())
    xx = np.column_stack((x, y))
    xy = xx[xx[:,0].argsort()]
    ax.plot(xy[:,0], xy[:,1])
    ax.set_xlabel("Hz")
    ax.set_ylabel("KL - Div")
    ax.set_xticks(xy[:,0])
    # ax.scatter(x, y, 'x')
    ax.set_title("KL-Div {}: {}".format(title, depth))

def plotExperimentSummary(directory, exp_list, output, depth, title="", save=False):
    no_exp = len(exp_list) + 1        # + 1 for the KL div plot
    rs = np.ceil(np.sqrt(no_exp)).astype(np.int)
    cls = np.ceil(no_exp / rs).astype(np.int)
    fig, axs = plt.subplots(rs, cls, figsize=(2*15, 2*9))
    tree_dict = {}
    # Plotting all the trees
    for i, f in enumerate(exp_list):
        fname = os.path.join(directory, f)
        exp = regexFilename(f)
        c = i % cls
        r = i // cls
        t = qt.Quadtree.load(fname)
        print("loading tree {} done. proceeding with pruning".format(exp))
        t.postprocess(depth=depth)
        # Getting the part for the KL-DIV out
        skip = int(exp.split("-")[-1])
        fr = getFreq(skip, title)
        tree_dict[fr] = t
        # done with KL-DIv dictionary
        print("pruning tree {} done. Proceeding with plotting".format(exp))
        t.plot_tree(axs[r,c])
        axs[r,c].set_title("{} - {} Hz".format(exp, fr))

    # Calculating the kL DIV
    print("Done with plotting. Getting the KL Divrgence")
    base_freq = 8                # should always be 8! because that's the Hz we wanted
    base_tree = tree_dict[base_freq]
    del tree_dict[base_freq]
    kl_dict = {}
    for k, v in tree_dict.items():
        # TODO: find a way to put the perc_used and full into each axis - work with the title and the key?
        perc_used, full, comp_val = compareTwoTrees(base_tree, tree_comp=v, idx=k, d=depth)
        kl_dict[k] = comp_val
    # plotting the KL-Div in the next element
    c = len(exp_list)  % cls
    r = len(exp_list)  // cls
    plot_kl_dict(kl_dict, axs[r,c], title, depth)
    
    plt.suptitle("Experiment: {}, Depth: {}".format(title, depth))
    plt.tight_layout()
    if save:
        fn = os.path.join(output,"{}-{}".format(title, depth))
        print("Saving figure to: {}".format(fn))
        plt.savefig(fn, bbox_inches="tight")
    else:
        plt.show()


def compareTwoTrees(tree_base : qt.Quadtree, tree_comp: qt.Quadtree, idx, d):
    """
        ! Base SHOULD be a superclass of the other tree - because the other tree just SKIPS these observations -> assume that the same indices are in there. So iterate over that set of indices and check if they are present in the other dictionary
        ! add the keys that have been checked to a set, so that we can later check how many are NOT in the tree.
    """
    inters = tree_base.dictionary.keys() & tree_comp.dictionary.keys()
    full = tree_base.dictionary.keys() | tree_comp.dictionary.keys()
    not_inters = full - inters
    # print("For {} hz and depth {}, the full length of indices is {}, the intersection is {} and the not intersection is {}".format(idx, d, len(full), len(inters), len(not_inters)))
    kl = 0
    for k in inters:
        base = tree_base[k].getprobabilities()
        comp = tree_comp[k].getprobabilities()
        l = 1 #tree_base.getlevel(k)
        kl += kl_div(base, comp, weight=1/l)
    perc_used = len(inters) / len(full)
    return perc_used, len(full) ,kl


def regexFilename(name):
    out = re.search(r'qt-(\d{1,2})',name)
    return out.group()

def getFreq(skip, title):
    di = getBaseFreq(title)
    return int(di / skip)

def getBaseFreq(title):
    if "exp" in title:
        return 24
    else:
        return 32

def kl_div(vec_true, vec_pred, weight=1):
    """
    use the definition of the KL divergence
        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        * look at octomap paper for their defintion - the same!
    """
    kl = np.sum(np.log(vec_true / vec_pred) * vec_true)
    return weight * kl
    # negative cross-entropy
    # return np.sum(vec_true*np.log(vec_pred)) * (-1.0)

def parse_args(defaultdir):
    """
        Function to parse the arguments 
    """
    parser = ArgumentParser(description="Plotting two quadtrees that have been generated by the same file.")
    parser.add_argument("-d", "--depth", help="Plotting depth of the quadtree. Default is 4.", type=int, default=4)
    parser.add_argument("--input", help="Input directory. Default is the one of the file", type=str, default=defaultdir)
    parser.add_argument("--file", help="Input Experiment. Used to read the file which contains the string given by file from input directory. If not specified will raise FileNotFoundError", type=str, default="sim_tgt1-descend")
    parser.add_argument("-s", "--save", help="Whether to save or plot the diagram", action="store_true", default=False)
    parser.add_argument("--output", help="Output directory where to save the figures if save is set. Default is input + imgs", default=os.path.join(defaultdir, "imgs"))
    args = parser.parse_args()
    return args

if __name__=="__main__":
    thisdir = os.path.dirname(__file__)
    # a = datetime(2021, 10, 8)
    # outdir_date = a.strftime("%y-%m-%d")
    defdir = os.path.join(thisdir, '..', 'output', "skips")
    args = parse_args(defdir)
    
    f = findexp(args.file, args.input)
    # compareexp(args.input, f, args.file, depth=args.depth)
    plotExperimentSummary(directory=args.input, exp_list = f, output=args.output, depth = args.depth, title=args.file, save=args.save)
    # cform = c.strftime("%y-%m-%d")
    # dirtoplot = os.path.abspath(os.path.join(thisdir, '..', 'output', cform))
    # plotdir(dirtoplot)
    
    print("Plotting done")