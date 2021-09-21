#!/usr/bin/env python3

"""
    File to write a hashed implementation of a Quadtree and display it.
"""
import matplotlib.pyplot as plt
import quadtree as quadt
import numpy as np
from matplotlib.patches import Rectangle

############################################
# Not QT-tree class functions following here
############################################
def testBox(scale=100):
    """
        Function to test whether the box returning algorithm works. 
    """
    qt = quadt.Quadtree(scale=scale)
    fig, ax = plt.subplots()
    ax.scatter(50, 50)

    b = qt.getBox(15)
    lo, w, h = b.matplotlib_format()
    r = Rectangle(lo, w, h)
    plt.xlim(0, scale)
    plt.ylim(0, scale)
    ax.add_patch(r)
    plt.show()

def testInsert(scale=100):
    """
        Function to test whether inserting a point works.
        And how to display the entire box tree
    """
    x = 10
    y = 99
    qt = quadt.Quadtree(scale=scale, max_depth=6)
    fig, ax = plt.subplots()
    idx = qt.insert_point(x, y)
    idx = qt.insert_point(y, x)
    ax.scatter(x, y)
    ax.scatter(y, x)
    bxs = qt.getallboxes()
    print("Maximum Depth: {}".format(qt.max_depth))
    title = "Index: {}.".format(idx)
    for k, v in bxs.items():
        title += " {}".format(k)
        lo, w, h = v.matplotlib_format()
        r = Rectangle(lo, w, h, facecolor='none', edgecolor='red', lw=2)
        ax.add_patch(r)
    plt.xlim(0, scale)
    plt.ylim(0, scale)
    plt.title(title)
    plt.show()

def calculateMaxArea(max_number_of_nodes_abzissa=32):
    qt = quadt.Quadtree()
    max_ls = 1
    nodes = 1
    max_number_of_nodes = 2**max_number_of_nodes_abzissa
    print("Maximum nodes at {} bit are: {}".format(max_number_of_nodes_abzissa, max_number_of_nodes))
    while nodes < max_number_of_nodes:
        max_ls += 1
        nodes = qt.getMaxBoxes(max_ls)
        print("Number of nodes up to level: {} are {}".format(max_ls, nodes))
    max_ls -= 1
    nodes = qt.getMaxBoxes(max_ls)
    print("Maximum number of levels: {}".format(max_ls))
    nodes_before = qt.getMaxBoxes(max_ls - 1)
    
    nodes_last_level = nodes - nodes_before
    print("Number of nodes in the last level: {}".format(nodes_last_level))
    print("Along one axis, that is equivalent to: {} and at 1km2 in the beginning a resolution of: {} mm".format(np.sqrt(nodes_last_level), 1000000 / nodes_last_level))

def generatePoints(bounds=(0,100), n=5):
    sample_x = np.random.uniform(bounds[0], bounds[1], n)
    sample_y = np.random.uniform(bounds[0], bounds[1], n)
    # pts = np.asarray([sample_x, sample_y]).T
    vals = np.random.random((n, 3)) # generating random colours
    pt_dict = {}
    for i in range(n):
        pt_dict[(sample_x[i], sample_y[i])] = vals[i,:]
    return pt_dict

def test_points_pruning(qt : quadt.Quadtree):
    pts = generatePoints()
    idcs = qt.find_idcs(pts)
    fig, axs = plt.subplots(1,2)
    print(idcs)
    qt.insert_points(idcs)
    qt.printMotherChain(idcs)
    qt.printvals()
    # And plot the tree
    axs[0].scatter(pts[:,0], pts[:,1])
    axs[1].scatter(pts[:,0], pts[:,1])
    qt.plot_tree(axs[0])
    axs[0].set_xlim(0, 100)
    axs[0].set_ylim(0, 100)
    axs[1].set_xlim(0, 100)
    axs[1].set_ylim(0, 100)
    # plt.show()
    # Then prune the tree
    qt.prune(list(idcs))  # TODO: work on the pruning step

    # Then display the tree
    qt.plot_tree(axs[1])
    axs[1].set_title("Pruned")
    plt.show()
    print("testline")

def test_index_reduction(qt : quadt.Quadtree) -> None:
    pt_dict = generatePoints()
    idcs_dict = qt.find_idcs(pt_dict)
    print("First set of points:\n{}".format(idcs_dict))
    # Test the set of indices whether they should be inserted at a higher level. Insert at higher levels if None of the siblings is in the current set. 
    reduced_idcs_dict = qt.reduce_idcs(idcs_dict)
    print("Reduced indices:\n{}".format(reduced_idcs_dict))
    qt.insert_points(reduced_idcs_dict)
    qt.printvals()
    # Plotting
    fig, axs = plt.subplots(1,2)
    pts = np.asarray(list(pt_dict.keys()))
    axs[0].scatter(pts[:,0], pts[:,1])
    axs[1].scatter(pts[:,0], pts[:,1])
    qt.plot_tree(axs[0])
    axs[0].set_xlim(0, 100)
    axs[0].set_ylim(0, 100)
    axs[0].set_title("First set of points")

    pt_dict = generatePoints()
    idcs_dict = qt.find_idcs(pt_dict)
    print("Second set of points:\n{}".format(idcs_dict))
    reduced_idcs_dict = qt.reduce_idcs(idcs_dict)
    print("Reduced indices:\n{}".format(reduced_idcs_dict))
    qt.insert_points(reduced_idcs_dict)
    qt.plot_tree(axs[1])
    pts = np.asarray(list(pt_dict.keys()))
    axs[1].scatter(pts[:,0], pts[:,1])
    axs[1].set_xlim(0, 100)
    axs[1].set_ylim(0, 100)
    axs[1].set_title("Second set of points")
    # qt.prune(list(idcs))
    # qt.plot_tree(axs[1])
    plt.show()

def test_four_cycles(qt : quadt.Quadtree) -> None:
    plots = 3
    fig, axs = plt.subplots(3,plots)
    for i in range(3 * plots):
        test_single_cycle(qt, axs, i, plots)
    plt.show()

def test_single_cycle(qt : quadt.Quadtree, axs : plt.Axes, idx : int, width : int) -> None:
    c = idx % width
    r = idx // width
    pt_dict = generatePoints()
    idcs_dict = qt.find_idcs(pt_dict)
    print("{} set of points: {}".format(idx+1, idcs_dict))
    reduced_idcs_dict = qt.reduce_idcs(idcs_dict)
    print("Reduced indices:\n{}".format(reduced_idcs_dict))
    qt.insert_points(reduced_idcs_dict)
    disp_pts = np.asarray(list(pt_dict.keys()))
    axs[r, c].scatter(disp_pts[:,0], disp_pts[:,1])
    qt.plot_tree(axs[r,c])
    axs[r, c].set_xlim(0, 100)
    axs[r, c].set_ylim(0, 100)
    axs[r, c].set_title("{} points".format(idx+1))


def test_depth_calc(qt : quadt.Quadtree):
    """
        function to test the depth calculation
    """
    # test_vals = [1, 2, 5, 9, 10, 12, 13, 15, 21, 22, 80, 100, 123456]
    mb = quadt.Quadtree.getMaxBoxes(l=12)
    test_vals = np.random.triangular(0,40, mb, size=20)
    for val in test_vals:
        val = np.floor(val)
        l_log = qt.getlevel_log(val)
        l = qt.getlevel(val)
        print("For value {} the level calculated by the log is: {} and calculated by formula is: {}".format(
            val, l_log, l
        ))

def testlogodds():
    print("Function to test the log odds")
    init_prior = np.asarray([0.5, 0.5])
    logprior = np.log((init_prior / (1. - init_prior)))
    sens_model = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    log_model = np.log( sens_model / (1. - sens_model))
    print("initial prior:\n{}".format(init_prior))
    print("Log prior:\n{}".format(logprior))
    print("Sensor Model:\n{}".format(sens_model))
    print("Log Sensor model:\n{}".format(
        log_model
    ))

    cts_obs = np.zeros((10, init_prior.shape[0]))
    cts_obs[:] = logprior

    for i in range(10):
        print("Iteration: {}".format(i))
        true_state = 1
        observation = generate_sample(true_state, sens_model)
        print("True state is: {}, Generated observation is: {}".format(true_state, observation))
        prior = cts_obs[i,:]
        print("Prior is: {}".format(prior))
        log_n = updatebelief(observation, log_model, prior,  logprior)
        print("Updated log belief is: {}".format(log_n))
        cts_obs[i+1, :] = log_n
        bel = getbelief(log_n)
        print("Updated belief is: {}".format(bel))



def generate_sample(idx, prob_model):
    sam = np.random.choice(prob_model.shape[0], 1, p=prob_model[:,idx])
    return sam

def updatebelief(sample, model, prior, init_belief):
    obs = np.squeeze(model[:,sample])
    newbel = prior + obs - init_belief
    return newbel

def getbelief(log_odds):
    return (1. - (1. / ( 1. + np.exp(log_odds))))

if __name__=="__main__":
    np.random.seed(20)
    outside_bounds = (100, 100)
    # qt = quadt.Quadtree(scale=outside_bounds[0], max_depth=4)
    # test_depth_calc(qt)
    # test_four_cycles(qt)
    # testdict = {1: "val1", 2: "val2", 3: "val3", 4:"val4"}
    # a = iter(testdict)
    # b = next(a)

    testlogodds()

    # print(b)
    # while a:
    #     print(next(a))
    # print(next(iter(testdict)))
