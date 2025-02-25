#!/usr/bin/env python

"""
    Following this example:
    https://www.programcreek.com/python/?code=uzh-rpg%2Frpg_trajectory_evaluation%2Frpg_trajectory_evaluation-master%2Fscripts%2Fdataset_tools%2Fbag_to_pose.py

    Inserting into two trees - similar to what the octomap people proposed
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
    parser.add_argument("-r", "--rate", default=False, action="store_true", help="If parameter rate is set, the frequency will be limited to roughly 8 hz - ergo some PCLs will be dropped")
    parser.add_argument("-cr", "--comparison", default=2, help="The x-th value, which gets inserted into the second tree. By modulo operation", type=int)
    args = parser.parse_args()
    return args

def find_dimensions(directory, experiment):

    fname = os.path.expanduser(os.path.join(directory, experiment+".csv"))
    df = pd.read_csv(fname)
    min_x = df.at[0,"min_x"]
    min_y = df.at[0, "min_y"]
    max_x = df.at[0, "max_x"]
    max_y = df.at[0, "max_y"]
    x_scale = max_x - min_x - 1.
    y_scale = max_y - min_y - 1.
    scale = max(x_scale, y_scale) + 1
    return int(min_x), int(min_y), int(scale)


def getskiprate(fname):
    """
        Returns the image which is to be processed.
        *  Experiments ran at 24 Hz, so 3 will decrease to 8 Hz
        *  Simulations ran at 32 Hz, so 4 will decrease to 8 Hz
    """
    if "exp" in fname:
        return 3
    else:
        return 4

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
    lowx, lowy, scale = find_dimensions(args.input, args.file)
    low = (lowx, lowy)

    # comparison value
    compar = args.comparison

    tree_1 = qt.Quadtree(low=low, scale=scale, max_depth=max_depth)
    tree_2 = qt.Quadtree(low=low, scale=scale, max_depth=max_depth)
    status_statement = "Starting Dual simulation for {} with parameters: low {}, scale {}, depth {}. Putting every {} observation into the second tree".format(
        args.file, low, scale, max_depth, compar
    )

    # Rate delimiting setup
    if args.rate:
        rate = getskiprate(args.file)
        status_statement += " only processing every {} pcl".format(rate)
    print(status_statement)

    # symlinked - may have to use "realpath" or something
    bagf = os.path.expanduser(os.path.join(args.input, args.file + ".bag"))

    # rosbag setup
    pcl_topic = "/pcl_plane"
    ctr = 0

    
    ctr = 0 
    try:
        with rosbag.Bag(bagf, 'r') as bag:
            duration = bag.get_end_time() - bag.get_start_time()

            for (topic, msg, ts) in tqdm(bag.read_messages(topics=pcl_topic), desc='progress', total=bag.get_message_count(topic_filters=pcl_topic)):
                ctr += 1
                if args.rate:
                    # only process every X-th image.
                    if ctr % rate != 0:
                        continue
                # Decoding the points
                intensity_channel = msg.channels[1].values
                pts = msg.points
                val_dict = decode_intensity(pts, intensity_channel)
                
                # Inserting the points
                if ctr % compar == 0:
                    idcs = tree_2.insertion_idcs(pts, img_width, img_height)
                    tree_2.insert_points_arr(values=val_dict.values(), idcs = idcs)
                else:
                    idcs = tree_1.insertion_idcs(pts, img_width, img_height)
                    tree_1.insert_points_arr(values=val_dict.values(), idcs = idcs)

        # output saving
        # a = datetime.now()
        # d = a.strftime("%y-%m-%d_%H-%M")
        f1 = "{}-qt{}-{}-{}-{}-{}.pkl".format(args.file, 1, compar, max_depth, low, scale)
        f2 = "{}-qt{}-{}-{}-{}-{}.pkl".format(args.file, 2, compar, max_depth, low, scale)
        outpath1 = os.path.expanduser(os.path.join(args.output, f1))
        outpath2 = os.path.expanduser(os.path.join(args.output, f2))
        msg = tree_1.save(outpath1)
        tree_2.save(outpath2)
        print(msg)
    except FileNotFoundError: raise