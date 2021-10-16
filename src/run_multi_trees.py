#!/usr/bin/env python

"""
    File to launch the simulation in a new process for the PCL_Quadmaps.
"""


import subprocess
import os
import argparse
from datetime import datetime
from tqdm import tqdm

def get_experiments():
    # TODO: define by hand
    experiments = [
        "sim_tgt1-ascend", "sim_tgt1-descend", "sim_tgt2-ascend", "sim_tgt2-descend", "sim_mission-10m", "sim_mission-20m",
        "exp_tgt1-ascend", "exp_tgt1-descend", "exp_tgt2-ascend", "exp_tgt2-descend", "exp_hyb-freq-20m", "exp_mission-20m",     # missing  - exp_mission-10m
        "sim_hyb-10", "sim_hyb-20"
        ]
    return experiments

def getdepths():
    depths = [16, 14, 12, 10, 8, 6, 4, 1]
    return depths

if __name__=="__main__":
    cwd = os.path.abspath(os.path.dirname(__file__))
    tgt_fname = "load_multi_tree.py"

    # Directory and file management
    inputdir = os.path.abspath(os.path.join(cwd, '..', 'output', 'hpc', 'skip'))

    # Experiments and depths
    depths= getdepths()
    experiments = get_experiments()

    tgt_f = os.path.join(cwd, tgt_fname)
    print("Full length of experiments is: {}".format(len(experiments) * len(depths)))
    for e in tqdm(experiments):
        print("Launching files for experiment: {}".format(e))
        for d in depths:
            launchlist = ["python", tgt_f, "--input", inputdir, "--file", e ,"-d", str(d), "-s"]
            try:
                # print("Launching: {}".format(launchlist))
                print("Launching depth {} for experiment {}".format(d, e))
                subprocess.run(launchlist)
            except Exception as e:
                print(" failed for: {}: {}".format(launchlist, e))