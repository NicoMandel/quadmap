import quadtree as qt
import os.path
import matplotlib.pyplot as plt

from datetime import datetime

if __name__=="__main__":
    thisdir = os.path.dirname(__file__)
    outputdir = os.path.abspath(os.path.join(thisdir, '..', 'output'))

    a = datetime(2021, 10, 4, 10, 20)
    d = a.strftime("%y-%m-%d_%H-%M")
    experiment = "sim_tgt2-descend.bag"
    max_depth = 16
    low = (-80,-80)
    scale = 120
    
    outputdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
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