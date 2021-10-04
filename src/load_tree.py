import quadtree as qt
import os.path
import matplotlib.pyplot as plt

from datetime import datetime

if __name__=="__main__":
    thisdir = os.path.dirname(__file__)

    a = datetime(2021, 10, 4, 16, 45)
    d = a.strftime("%y-%m-%d_%H-%M")
    experiment = "sim_tgt1-descend"
    max_depth = 16
    low = (-40,-40)
    scale = 100
    
    outputdir = os.path.abspath(os.path.join(thisdir, '..', 'output', 'sim'))
    f = "{}_{}-qt-{}-{}-{}.pkl".format(d, experiment, max_depth, low, scale)
    
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