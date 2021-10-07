#!/usr/bin/env python

"""
    File to launch the simulation in a new process for the PCL_Quadmaps.
"""


import subprocess
import os
import argparse
from datetime import datetime

def get_filenames(path):
    fs = os.listdir(path)
    base = [os.path.splitext(f)[0] for f in fs]
    return base

def getscales(modename):
    """
        Function to get lowx and lowy and scale depending on the name of the experiment. FOr now: all the same:
        lowx = -90
        lowy = -30
        scale = 250
        from: exp_hyb_20
        ! defaults thrown into the other file
    """
    lowx = -90
    lowy = -30
    scale = 250
    return lowx, lowy, scale

if __name__=="__main__":
    cwd = os.path.abspath(os.path.dirname(__file__))
    tgt_fname = "bag-test.py"

    # Directory and file management
    inputdir = os.path.abspath(os.path.expanduser("~/rosbag/pcl"))
    mode = get_filenames(inputdir)
    outputdir = os.path.abspath(os.path.join(cwd, '..', 'output', 'sim'))

    a = datetime(2021, 10, 6, 18, 00)
    # d = a.strftime("%y-%m-%d_%H-%M")
    # a = datetime.now()
    d = a.strftime("%y-%m-%d_%H-%M")
    outdir = os.path.join(outputdir, d)
    # os.mkdir(outdir)

    tgt_f = os.path.join(cwd, tgt_fname)
    # base_launch = "python {} ".format(tgt_f)            # Full path
    for m in mode:
        # run first with default values
        # launchstring = base_launch + "--input {} --file {} --output {}".format(inputdir, m, outputdir)
        # lowx, lowy, scale = getscales(m)
        launchlist = ["python", tgt_f, "--input", inputdir, "--file", m, "--output", outdir, "-r"] #, "-lx" , lowx, "-ly", lowy, "-sc", scale]
        try:
            print("Launching: {}".format(launchlist))
            subprocess.Popen(launchlist)
        except Exception as e:
            print("Simulation failed for: {}: {}".format(launchlist, e))