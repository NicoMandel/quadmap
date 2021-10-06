import quadtree as qt
import os.path
import matplotlib.pyplot as plt

from datetime import datetime

if __name__=="__main__":
    thisdir = os.path.dirname(__file__)

    a = datetime(2021, 10, 6, 12, 23)
    d = a.strftime("%y-%m-%d_%H-%M")
    b = datetime(2021, 10, 6, 13, 8).strftime("%y-%m-%d_%H-%M")
    experiment = "exp_tgt1-descend"
    max_depth = 16
    low = (-40,-40)
    scale = 100
    
    outputdir = os.path.abspath(os.path.join(thisdir, '..', 'output', 'sim', d))
    f = "{}_{}-qt-{}-{}-{}.pkl".format(b, experiment, max_depth, low, scale)
    
    fname = os.path.join(outputdir,f)
    tree = qt.Quadtree.load(fname)
    print("Loading done. Proceeding with pruning")
    tree.postprocess()

    print("Pruning done, Proceeding with Plotting")

    fig = plt.figure(figsize=(15,9))
    ax = fig.gca()
    tree.plot_tree(ax)

    plt.show()
    print("Plotting done")