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

def clean_filelist(flist: list) -> dict:
    """
        clearning a list of file names. Returns a dictionary with the filenames as keys and the parameters in a tuple
    """
    flist_dict = {}
    for f in flist:
        flist_dict[f] = clean_filename(f)
    return flist_dict

def clean_filename(f: str):
    """
        Getting the parameters out of a name. Gets:
        * sim / exp
        * mission mode
        * hz
    """
    sim = f.split("_")[0]
    # regex to "everything before"
    # out = re.search(r'^(.*?)\-qt-(\d{1,2})',f.split("_")[1])
    out2 = re.search(r'^.*(?=(\-qt-(\d{1,2})))', f.split("_")[1])
    mm = out2.group()
    # Splitting up into "mode" and "experiment type"
    # ? potentially use later - currently not 100 % necessary
    # mmm = mm.split("-")
    # exp_type = mmm[0]
    # exp_setting = mmm[1]
    # mode = out.group()
    # Converting the frequency
    out = re.search(r'qt-(\d{1,2})', f.split("_")[1])
    hz = out.group().split("-")[1]
    hz = getFreq(int(hz),sim)
    return sim, mm, hz


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

def plot_kl_dict(kl_dict, ax, col1_title, col2_title, figtitle, depth):
    
    # ndict = sorted(kl_dict)
    # x = np.asarray(list(ndict.keys()))
    # y = np.asarray(list(ndict.values()))
    x = list(kl_dict.keys())
    y = np.asarray(list(kl_dict.values()))
    xx = np.column_stack((x, y))
    xy = xx[xx[:,0].argsort()]
    p1 = ax.plot(xy[:,0], xy[:,1], label=col1_title)
    ax.tick_params(axis='y', labelcolor=p1[0].get_color())
    ax.set_xlabel("Hz")
    ax.set_ylabel("KL - Div weighted")
    ax.set_xticks(xy[:,0])
    # add a new axis object
    ax1 = ax.twinx()
    ax1.plot(xy[:,0], xy[:,2], label=col2_title, color='red')
    ax1.set_ylabel("KL - Div unweighted")
    ax1.tick_params(axis='y', labelcolor='red')

    # ax.scatter(x, y, 'x')
    # ax1.legend()
    # ax.legend()
    ax.set_title("KL-Div {}: {}".format(figtitle, depth))    

def plotExperimentSummary(directory, exp_list, output, depth, suptitle="", save=False):
    no_exp = len(exp_list) + 1        # + 1 for the KL div plot
    rs = np.ceil(np.sqrt(no_exp)).astype(np.int)
    cls = np.ceil(no_exp / rs).astype(np.int)
    fig, axs = plt.subplots(rs, cls, figsize=(2*15, 2*9))
    tree_dict = {}
    # First filling in the dictionary and calculating the KL-div. Then plotting.
    # Plotting all the trees
    for i, f in enumerate(exp_list):
        fname = os.path.join(directory, f)
        exp = regexFilename(f)
        t = qt.Quadtree.load(fname)
        print("loading tree {} done. proceeding with pruning".format(exp))
        t.postprocess(depth=depth)
        # Getting the part for the KL-DIV out
        skip = int(exp.split("-")[-1])
        fr = getFreq(skip, suptitle)
        tree_dict[fr] = t

    # Calculating the KL-divergence
    base_freq = 8                # should always be 8! because that's the Hz we wanted
    base_tree = tree_dict[base_freq]
    del tree_dict[base_freq]
    kl_dict = {}
    kl_stat_dict = {}

    for k, v in tree_dict.items():
        perc_used, full, comp_weighted = compareTwoTrees(base_tree, tree_comp=v, weighted=True)
        _, _, comp_unweighted = compareTwoTrees(base_tree, tree_comp=v, weighted=False)
        kl_dict[k] = (comp_weighted, comp_unweighted)
        # kl_unweighted_dict[k] = comp_unweighted
        kl_stat_dict[k] = (perc_used, full)

    # Plotting section
    for i, k in enumerate(sorted(tree_dict.keys())):
        c = i % cls
        r = i // cls
        tree_dict[k].plot_tree(axs[r,c])
        stat = kl_stat_dict[k]
        title = "{} Hz, {:.1f} % of {} used for comparison".format(k, stat[0]*100, stat[1])
        axs[r,c].set_title(title)
    # Calculating the kL DIV
    print("Done with plotting. Getting the KL Divergence")
    # Plotting the base tree
    c = (len(exp_list) - 1) % cls
    r = (len(exp_list) - 1)  // cls
    base_tree.plot_tree(axs[r,c])
    axs[r,c].set_title("{} Hz".format(base_freq))

    # plotting the KL-Div in the next element
    c = len(exp_list)   % cls
    r = len(exp_list)   // cls
    plot_kl_dict(kl_dict, axs[r,c], "weighted", "unweighted", suptitle, depth)
    
    plt.suptitle("Experiment: {}, Depth: {}".format(suptitle, depth))
    plt.tight_layout()
    if save:
        fn = os.path.join(output,"{}-{}".format(suptitle, depth))
        print("Saving figure to: {}".format(fn))
        plt.savefig(fn, bbox_inches="tight")
    else:
        plt.show()


def plotCases(fdict : dict, compare_cases : list, directory, depth):
    """
        Function to plot three cases on top of each other
    """
    # Setting matplotlib parameters
    # font ={'family' : 'sans-serif',
    #         'weight' : 'bold',
    #         'size'  : 22}
    plt.rcParams.update({'font.family': 'sans-serif', 'font.weight' : 'bold' , # 'font.size': 20,
                        'axes.titlesize': 'x-large', 'axes.labelsize' : 'large', 'axes.titleweight' : "bold",
                        'figure.titleweight' : 'bold', 'figure.titlesize': 'xx-large'})
    #

    tgt_dict  = {}
    for cs in compare_cases:
        for k, v in fdict.items():
            if cs == v:
                tgt_dict[v] = k
    tree_dict = {}
    for k,v in tgt_dict.items():
        fname = os.path.join(directory, v)
        tree = qt.Quadtree.load(fname)
        tree.postprocess(depth=depth)
        tree_dict[k] = tree
    print("Tree postprocessing done. Continuing with plotting")
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(6,12))
    for i, cs in enumerate(tree_dict.keys()):
        tree = tree_dict[cs]
        tree.plot_tree(axs[i])
        # csspl = cs[1].split("-")[1]
        tit = "{} {} Hz".format(cs[1], cs[2])
        axs[i].set_title(tit, fontname="Times New Roman")

        # zooming - sim tgt 1
        # axs[i].set_xlim((-27,-17))
        # axs[i].set_ylim((15,25))
        # axs[i].set_xticks([-22])
        # axs[i].set_yticks([20])

        # sim - tgt2
        # axs[i].set_xlim((-50,-40))
        # axs[i].set_ylim((25,35))
        # axs[i].set_xticks([-45])
        # axs[i].set_yticks([30])

        # sim - hyb 20 - tgt2
        # axs[i].set_xlim((-50,-35))
        # axs[i].set_ylim((20,35))
        # axs[i].set_xticks([-45])
        # axs[i].set_yticks([25])


        # exp - tgt 1
        # axs[i].set_xlim((-15,-5))
        # axs[i].set_ylim((-5,5))
        # axs[i].set_xticks([-10])
        # axs[i].set_yticks([0])

        # # exp - tgt2
        # axs[i].set_xlim((-25,-15))
        # axs[i].set_ylim((10,20))
        # axs[i].set_xticks([-20])
        # axs[i].set_yticks([15])

        # sim - hyb 20
        # axs[i].set_xlim((-50,-20))
        # axs[i].set_ylim((10,40))
        # axs[i].set_xticks([-22, -45])
        # axs[i].set_yticks([20, 30])

        # # exp - hyb 20
        # axs[i].set_xlim((-12,-2))
        # axs[i].set_ylim((-5,5))
        # axs[i].set_xticks([-7.5])
        # axs[i].set_yticks([0])


        axs[i].patch.set_visible(False)


        [axs[i].spines[loc].set_visible(False) for loc in ["top", "right", "bottom", "left"]]

    # Setting the background to transparent:
    fig.patch.set_visible(False)
    
    fig.suptitle("{}-{}".format(compare_cases[0][0], depth))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.subplots_adjust(top=0.85)
    plt.show()
    print("Plotting Done")


def compareTwoTrees(tree_base : qt.Quadtree, tree_comp: qt.Quadtree, weighted=False):
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
        if weighted:
            l = tree_base.getlevel(k)
        else:
            l = 1.
        kl += kl_div(base, comp, weight=1./l)
    perc_used = len(inters) / len(full)
    return perc_used, len(full), kl


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
    parser.add_argument("-d", "--depth", help="Plotting depth of the quadtree. Default is 4.", type=int, default=1)
    parser.add_argument("--input", help="Input directory. Default is the one of the file", type=str, default=defaultdir)
    parser.add_argument("--file", help="Input Experiment. Used to read the file which contains the string given by file from input directory. If not specified will raise FileNotFoundError", type=str, default="exp_tgt1")
    parser.add_argument("-s", "--save", help="Whether to save or plot the diagram", action="store_true", default=False)
    parser.add_argument("--output", help="Output directory where to save the figures if save is set. Default is input + imgs", default=os.path.join(defaultdir, "imgs"))
    args = parser.parse_args()
    return args

if __name__=="__main__":
    thisdir = os.path.dirname(__file__)
    a = datetime(2021, 10, 11)
    outdir_date = a.strftime("%y-%m-%d")
    defdir = os.path.join(thisdir, '..', 'output', 'hpc', outdir_date)
    args = parse_args(defdir)
    
    f = findexp(args.file, args.input)
    f_dict = clean_filelist(f)
    # for hybrid and missions
    if "tgt" not in args.file:
        # use hyb_20 - 8, hyb_20 - 1 and mission_20-8
        if "sim" in args.file:
            comparison_cases = [(args.file, "hyb-20", 8), (args.file, "hyb-20", 1), (args.file, "mission-20m", 8)]
        else:
            comparison_cases = [(args.file, "hyb-freq-20m", 8), (args.file, "hyb-freq-20m", 1), (args.file, "mission-20m", 8)]
    else:
        sim = args.file.split("_")[0]
        case = args.file.split("_")[1]
        # For target ascends and descends
        comparison_cases = [(sim, case+"-descend",8), (sim, case+"-descend",1), (sim, case+"-ascend",8)]

    

    plotCases(f_dict, comparison_cases, args.input, args.depth)
    # compareexp(args.input, f, args.file, depth=args.depth)
    # plotExperimentSummary(directory=args.input, exp_list = f, output=args.output, depth = args.depth, suptitle=args.file, save=args.save)
    # cform = c.strftime("%y-%m-%d")
    # dirtoplot = os.path.abspath(os.path.join(thisdir, '..', 'output', cform))
    # plotdir(dirtoplot)
    
    print("Plotting done")