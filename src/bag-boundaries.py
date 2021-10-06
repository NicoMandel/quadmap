#!/usr/bin/env python

"""
    Following this example:
    https://www.programcreek.com/python/?code=uzh-rpg%2Frpg_trajectory_evaluation%2Frpg_trajectory_evaluation-master%2Fscripts%2Fdataset_tools%2Fbag_to_pose.py
"""

import os.path
import rosbag
from datetime import datetime
from tqdm import tqdm

from argparse import ArgumentParser

import pandas as pd




def parse_args(defaultdir):
    """
        Function to parse the arguments 
    """
    parser = ArgumentParser(description="Stepping through a pcl bag file to insert into a quadmap to save it.")
    parser.add_argument("--input", help="Input directory. If not specified, will raise FileNotFoundError", type=str)
    parser.add_argument("--output", help="Output directory. Default is ../output", default=defaultdir, type=str)
    parser.add_argument("--file", help="Input Filename. Used to read input from input directory and create equivalent output", type=str)

    args = parser.parse_args()
    return args


def find_boundaries(pts, min_x, min_y, max_x, max_y):
    
    for pt in pts:
        if pt.x < min_x:
            min_x = pt.x
        elif pt.x > max_x:
            max_x = pt.x
        
        if pt.y < min_y:
            min_y = pt.y
        elif pt.y > max_y:
            max_y = pt.y
    
    return min_x, min_y, max_x, max_y



if __name__=="__main__":
    dirname = os.path.dirname(os.path.abspath(__file__))

    # argument parsing
    defaultdir = os.path.join(dirname, '..', 'output')
    args = parse_args(defaultdir)

    # symlinked - may have to use "realpath" or something
    bagf = os.path.join(args.input, args.file + ".bag")

    # rosbag setup
    pcl_topic = "/pcl_plane"
    ctr = 0

    # timing setup
    timerlist = []
    hz = 30
    min_x = min_y = 1000
    max_x = max_y = -1000
    
    try:
        with rosbag.Bag(bagf, 'r') as bag:
            duration = bag.get_end_time() - bag.get_start_time()

            for (topic, msg, ts) in tqdm(bag.read_messages(topics=pcl_topic), desc='progress', total=bag.get_message_count(topic_filters=pcl_topic)):

                # Decoding the points
                pts = msg.points
                min_x, min_y, max_x, max_y = find_boundaries(pts, min_x, min_y, max_x, max_y)
                
                # Inserting the points

        # output saving
        a = datetime.now()
        d = a.strftime("%y-%m-%d_%H-%M")
        df = pd.DataFrame(columns=["min_x", "min_y", "max_x", "max_y"])
        df.at[0, "min_x"] = min_x
        df.at[0, "min_y"] = min_y
        df.at[0, "max_x"] = max_x
        df.at[0, "max_y"] = max_y
        
        # TODO: change this fileformat
        f = "{}.csv".format(args.file)
        outpath = os.path.join(args.output, f)
        df.to_csv(outpath)
        print("Written csv to: {}".format(outpath))
    except FileNotFoundError: raise