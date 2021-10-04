#!/usr/bin/env python

"""
    File to launch the simulation in a new process for the PCL_Quadmaps.
"""


import subprocess
import os
import argparse

def get_filenames(path):
    fs = os.listdir(path)
    base = [os.path.splitext(f)[0] for f in fs]
    return base

if __name__=="__main__":
    cwd = os.path.abspath(os.path.dirname(__file__))
    tgt_fname = "bag-test.py"

    # Directory and file management
    inputdir = os.path.abspath(os.path.expanduser("~/rosbag/sim"))
    mode = get_filenames(inputdir)
    outputdir = os.path.abspath(os.path.join(cwd, '..', 'output', 'sim'))

    tgt_f = os.path.join(cwd, tgt_fname)
    # base_launch = "python {} ".format(tgt_f)            # Full path
    for m in mode:
        # run first with default values
        # launchstring = base_launch + "--input {} --file {} --output {}".format(inputdir, m, outputdir)
        launchlist = ["python", tgt_f, "--input", inputdir, "--file", m, "--output", outputdir]
        try:
            print("Launching: {}".format(launchlist))
            subprocess.Popen(launchlist)
        except Exception as e:
            print("Simulation failed for: {}: {}".format(launchlist, e))