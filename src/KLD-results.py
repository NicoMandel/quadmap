
import quadtree as qt
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
import re
import pandas as pd
from tqdm import tqdm

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

def plotExperimentSummary(directory : str, exp_dictionary: dict, output, depth : int, suptitle : str="" , save : bool =False, base_freq = 8):
    no_exp = len(exp_dictionary) + 1        # + 1 for the KL div plot
    fig = plt.figure(figsize=(15, 9))
    ax = fig.gca()
    settings_list = []
    hz_list = []
    for v in exp_dictionary.values():
        settings_list.append(v[1])
        hz_list.append(v[2])
    settings = set(settings_list)
    hz = set(hz_list)
    # Group experiments. do a dataframe.
    hz.remove(base_freq)
    df = pd.DataFrame(index=hz, columns=settings)
    stat_df = pd.DataFrame(index = hz, columns = settings)
    tree_dict = {}
    # First filling in the dictionary and calculating the KL-div. Then plotting.
    # Plotting all the trees
    asdfa = 0
    for setting in settings:
        # # ! quick breaker:
        # if asdfa == 1:
        #     break
        # asdfa += 1

        fnames = [k for k in exp_dictionary.keys() if setting in k]
        for f in fnames:
            fname = os.path.join(directory, f)
            t = qt.Quadtree.load(fname)
            print("loading tree {} done. proceeding with pruning".format(exp_dictionary[f]))
            t.postprocess(depth=depth)
            # Getting the part for the KL-DIV out
            fr = exp_dictionary[f][2]
            tree_dict[fr] = t

        # Calculating the KL-divergence
        base_tree = tree_dict[base_freq]
        del tree_dict[base_freq]

        for k, v in tree_dict.items():
            perc_used, full, comp_unweighted = compareTwoTrees(base_tree, tree_comp=v, weighted=False)
            df.at[k, setting] = comp_unweighted
            # kl_unweighted_dict[k] = comp_unweighted
            stat_df.at[k, setting] = (perc_used, full)

    # print(df)
    df_path = os.path.join(output, suptitle + "_kl_div.csv")
    df.to_csv(df_path)
    stat_df_path = os.path.join(output, suptitle + "_kl_div_stats.csv")
    stat_df.to_csv(stat_df_path)
    df.plot(ax=ax)
    plt.show()

def plotFromCsv(directory, name):
    """
        Potentially change the linestyles like this: https://stackoverflow.com/questions/14178194/python-pandas-plotting-options-for-multiple-lines
    """
    fpath1 = os.path.join(directory, "exp_" + name +".csv")
    df1 = pd.read_csv(fpath1, index_col=0)
    fpath2 = os.path.join(directory, "sim_" + name +".csv")
    df2 = pd.read_csv(fpath2, index_col=0)
    fig, ax = plt.subplots(1, 2, sharey=True)
    df1.plot(ax=ax[0], title="exp")
    df2.plot(ax=ax[1], title="sim")
    plt.tight_layout()
    plt.show()


def compareKLDSimilar(directory: str, fdict: dict, similarity : str, depth : int = 10):
    """
        Function to compare similar research. 
    """
    if "tgt" in similarity:
        sim = getSimilarity(similarity)
        tgt_dict = {k:v for k,v in fdict.items() if sim[0] in v[1]}
    else:
        tgt_dict = {k:v for k,v in fdict.items() if "tgt" not in v[1]}
        # dirty fix for comparing hyb-10 and hyb-20
        if "10" in similarity:
            tgt_dict = {k:v for k,v in fdict.items() if ("tgt" not in v[1]) and ("10" in v[1])}
        elif "20" in similarity:
            tgt_dict = {k:v for k,v in fdict.items() if ("tgt" not in v[1]) and ("20" in v[1])}
    
    # the baseline:
    basel_dict = getBaselineSim(tgt_dict, similarity)
    df = pd.DataFrame(index=tgt_dict.keys(), columns=["KL", "perc","full", "case", "motion", "freq"])
    basel_fname = list(basel_dict.keys())[0]
    basel_f = os.path.join(directory, basel_fname)
    basel_tree = qt.Quadtree.load(basel_f)
    basel_tree.postprocess(depth=depth)
    for k,v in tqdm(tgt_dict.items()):
        fname = os.path.join(directory, k)
        tree = qt.Quadtree.load(fname)
        tree.postprocess(depth=depth)
        perc_used, full, kl = compareTwoTrees(basel_tree, tree)
        df.at[k, "KL"] = kl
        df.at[k, "perc"] = perc_used
        df.at[k, "full"] = full
        df.at[k, "case"] = v[0]
        df.at[k, "motion"] = v[1]
        df.at[k, "freq"] = v[2]
    df.sort_values(by=["KL", "freq"], inplace=True)
    print(df)
    out_f = "_".join([basel_dict[basel_fname][0], basel_dict[basel_fname][1], str(depth)])
    out_fname = os.path.join(directory, out_f +".csv")
    print("Writing to {}".format(out_fname))
    df.to_csv(out_fname)



def getSimilarity(similarity : str) -> tuple:
    """
        Function to get the association of what things are considered similar
    """
    if "tgt" in similarity:
        if "tgt1" in similarity:
            tgt = "tgt1"
        else:
            tgt = "tgt2"
        return (tgt, "ascend", "descend")
    else:
        if "20" in similarity:
            hgt = 20
        elif "10" in similarity:
            hgt = 10
        return (hgt, "mission", "hyb")

def getBaselineSim(tgt_dict : dict, similarity : str):
    """
        Function to remove the baseline from the tgt_dict and return it
    """
    case = tgt_dict[list(tgt_dict.keys())[0]]
    base_freq = 8
    if 'scend' in case[1]:
        miss =  "-".join([similarity, "descend"])
    else:
        # TODO: fill this in
        miss = "hyb"
    base_tree_dict = {k: v for k, v in tgt_dict.items() if (miss in v[1]) and (base_freq == v[2])}
    del tgt_dict[list(base_tree_dict.keys())[0]]
    return base_tree_dict

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

def summarizecsv(dir):
    read_dir = os.path.join(dir, "csv")
    flist = [f for f in glob.glob(read_dir+ "/*.csv")]
    ddict = {}
    for f in flist:
        bname = os.path.basename(f).split(".")[0]
        df = pd.read_csv(f, index_col=0)
        ddict[bname] = df
    
    outfname = "summary.xlsx"
    outf = os.path.join(read_dir, outfname)
    with pd.ExcelWriter(outf) as writer:
        for k, df in ddict.items():
            df.to_excel(writer, sheet_name=k)

def celladjustKLD(directory, fname):
    f = os.path.join(directory, fname)
    xlf = pd.ExcelFile(f)
    dfd = {}
    for sheet_name in xlf.sheet_names:
        dfd[sheet_name] = xlf.parse(sheet_name, index_col=0)
    xlf.close()
    print("read the df")

    # make a new column
    for sheet, df in dfd.items():
        df["used_c"] = df["perc"] * df["full"]
        df["Weighted_KLD"] = df["KL"] / df["used_c"]
        df.sort_values(by=["Weighted_KLD", "freq"], inplace=True)
    
    outf = os.path.join(directory, "adjustedKLD.xlsx")
    with pd.ExcelWriter(outf) as writer:
        for sheet, df in dfd.items():
            df.to_excel(writer, sheet_name=sheet)

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
    parser.add_argument("-d", "--depth", help="Plotting depth of the quadtree. Default is 10.", type=int, default=16)
    parser.add_argument("--input", help="Input directory. Default is the one of the file", type=str, default=defaultdir)
    parser.add_argument("--file", help="Input Experiment. Used to read the file which contains the string given by file from input directory. If not specified will raise FileNotFoundError", type=str, default="sim")
    parser.add_argument("-s", "--save", help="Whether to save or plot the diagram", action="store_true", default=False)
    parser.add_argument("--output", help="Output directory where to save the figures if save is set. Default is this directory + ../output/hpc", default=defdir)
    parser.add_argument("--similarity", help="Which similarity should be evaluated", type=str, default="hyb-20")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    thisdir = os.path.dirname(__file__)
    a = datetime(2021, 10, 11)
    outdir_date = a.strftime("%y-%m-%d")
    defdir = os.path.join(thisdir, '..', 'output', 'hpc', outdir_date)
    args = parse_args(defdir)
    
    f = findexp(args.file, args.input)
    fdict = clean_filelist(f)
    # compareexp(args.input, f, args.file, depth=args.depth)
    # plotExperimentSummary(directory=args.input, exp_dictionary = fdict, output=args.output, depth = args.depth, suptitle=args.file, save=args.save)
    # plotFromCsv(args.output, "kl_div")
    # compareKLDSimilar(defdir, fdict, args.similarity, depth=args.depth)
    # summarizecsv(args.input)
    csvdir = os.path.join(defdir, "csv")
    celladjustKLD(csvdir, "summary-cleaned.xlsx")
    # cform = c.strftime("%y-%m-%d")
    # dirtoplot = os.path.abspath(os.path.join(thisdir, '..', 'output', cform))
    # plotdir(dirtoplot)
    
    print("Results done")