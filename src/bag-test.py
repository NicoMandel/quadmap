#!/usr/bin/env python

"""
    Following this example:
    https://www.programcreek.com/python/?code=uzh-rpg%2Frpg_trajectory_evaluation%2Frpg_trajectory_evaluation-master%2Fscripts%2Fdataset_tools%2Fbag_to_pose.py
"""

import os.path
import rosbag
from datetime import datetime
from tqdm import tqdm
import pandas as pd

import quadtree as qt

from argparse import ArgumentParser

def decode_intensity(pts, intensity):
        pt_dict = {}
        for i, ch in enumerate(intensity):
            pt = pts[i]
            pt_dict[tuple([pt.x, pt.y])] = int(ch)
        
        return pt_dict


def parse_args(defaultdir):
    """
        Function to parse the arguments 
    """
    parser = ArgumentParser(description="Stepping through a pcl bag file to insert into a quadmap to save it.")
    parser.add_argument("--width", default=128, help="Image width. Default is 128. Change depending on network resolution", type=int)
    parser.add_argument("--height", default=96, help="Image Height. default is 96. Change depending on network resolution", type=int)
    parser.add_argument("-d", "--depth", default=16, help="depth of the quadtree. Default is 16.", type=int)
    parser.add_argument("--input", help="Input directory. If not specified, will raise FileNotFoundError", type=str)
    parser.add_argument("--output", help="Output directory. Default is ../output", default=defaultdir, type=str)
    parser.add_argument("--file", help="Input Filename. Used to read input from input directory and create equivalent output", type=str)
    parser.add_argument("-r", "--rate", default=8, type=int, help="The rate, at which values should be inserted into the Quadtree. Will use integer divide for this")
    parser.add_argument("-t", "--take", help="Take the .csv file or use the values calculated by hand. To compare files with one another.", action="store_false", default=True)
    args = parser.parse_args()
    return args

def find_dimensions(directory, experiment, take_csv : bool):

    if take_csv:
        fname = os.path.expanduser(os.path.join(directory, experiment+".csv"))
        df = pd.read_csv(fname)
        min_x = df.at[0,"min_x"]
        min_y = df.at[0, "min_y"]
        max_x = df.at[0, "max_x"]
        max_y = df.at[0, "max_y"]
        x_scale = max_x - min_x -1.
        y_scale = max_y - min_y - 1.
        scale = max(x_scale, y_scale) + 1.
    else:
        if "sim" in experiment:
            if "tgt1" in experiment:
                min_x = -43
                min_y = -1
                scale = 41
            elif "tgt2" in experiment:
                min_x = -63
                min_y = 9
                scale = 40
            elif "mission" in experiment:
                min_x = -86
                min_y = -10
                scale = 90
            else: #hybrid
                min_x = -93
                min_y = -16
                scale = 94
        # experiments
        else:
            if "tgt1" in experiment:
                min_x = -31
                min_y = -21
                scale = 47
            elif "tgt2" in experiment:
                min_x = - 58
                min_y = - 11
                scale = 57
            elif "mission" in experiment:
                min_x = -72
                min_y = -31
                scale = 94
            else:
                min_x = -85
                min_y = -30
                scale = 255

    return int(min_x), int(min_y), int(scale)


def getskiprate(fname, r):
    """
        Returns the modulo which to divide by
        *  Experiments ran at ~ 24 Hz, so 3 will decrease to 8 Hz
        *  Simulations ran at ~ 32 Hz, so 4 will decrease to 8 Hz
    """

    if "exp" in fname:
        orig_rate= 24
    else:
        orig_rate= 32
    skiprate = orig_rate / r
    return int(skiprate)

if __name__=="__main__":
    dirname = os.path.dirname(os.path.abspath(__file__))

    # argument parsing
    defaultdir = os.path.join(dirname, '..', 'output')
    args = parse_args(defaultdir)

    # Quadtree values
    img_width = args.width
    img_height = args.height
    max_depth = args.depth

    # Getting the right dimensions
    lowx, lowy, scale = find_dimensions(args.input, args.file, args.take)
    low = (lowx, lowy)

    tree = qt.Quadtree(low=low, scale=scale, max_depth=max_depth)
    status_statement = "Starting Quadtree simulation for {} with parameters: low {}, scale {}, depth {}".format(
        args.file, low, scale, max_depth
    )

    # Rate delimiting setup
    rate = getskiprate(args.file, args.rate)
    status_statement += " only processing every {} pcl.".format(rate)
    print(status_statement)

    # symlinked - may have to use "realpath" or something
    bagf = os.path.expanduser(os.path.join(args.input, args.file + ".bag"))

    # rosbag setup
    pcl_topic = "/pcl_plane"
    ctr = 0

    try:
        with rosbag.Bag(bagf, 'r') as bag:
            duration = bag.get_end_time() - bag.get_start_time()

            for (topic, msg, ts) in tqdm(bag.read_messages(topics=pcl_topic), desc='progress', total=bag.get_message_count(topic_filters=pcl_topic)):
                ctr += 1
                # only process every X-th image.
                if ctr % rate == 0:                    
                    # Decoding the points
                    intensity_channel = msg.channels[1].values
                    pts = msg.points
                    val_dict = decode_intensity(pts, intensity_channel)
                    
                    # Inserting the points
                    idcs = tree.insertion_idcs(pts, img_width, img_height)
                    tree.insert_points_arr(values=val_dict.values(), idcs = idcs)
    except FileNotFoundError: raise

    try:
        # output saving
        f = "{}-qt-{}-{}-{}-{}.pkl".format(args.file, rate, max_depth, low, scale)
        outpath = os.path.expanduser(os.path.join(args.output, f))
        msg = tree.save(outpath)
        print(msg)
    except FileExistsError: raise