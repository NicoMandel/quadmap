import quadtree as qt
import os.path
import matplotlib.pyplot as plt

if __name__=="__main__":
    
    thisdir = os.path.dirname(__file__)
    outputdir = os.path.abspath(os.path.join(thisdir, '..', 'output'))
    fname = os.path.join(outputdir, 'qt_21-09-28_11-37.pkl')
    tree = qt.Quadtree.load(fname)
    print("Loading done. Proceeding with pruning")
    tree.postprocess()

    print("Pruning done, Proceeding with Plotting")

    fig = plt.figure(figsize=(15,9))
    ax = fig.gca()
    tree.plot_tree(ax)

    plt.show()
    print("Plotting done")