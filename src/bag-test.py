#!/usr/bin/env python

"""
    Following this example:
    https://www.programcreek.com/python/?code=uzh-rpg%2Frpg_trajectory_evaluation%2Frpg_trajectory_evaluation-master%2Fscripts%2Fdataset_tools%2Fbag_to_pose.py
"""

import os.path
import rosbag
from datetime import datetime
import time
import numpy as np
from tqdm import tqdm

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
    parser.add_argument("-lx", "--lowx", default=-40, help="lowest x value encountered in the pointcloud. Default is -40", type=int)
    parser.add_argument("-ly", "--lowy", default=-40, help="lowest y value encountered in the pointcloud. Default is -40", type=int)
    parser.add_argument("-sc", "--scale", default=100, help="scale of the Quadtree. Map extent will be low + scale. Default is 100", type=int)
    parser.add_argument("--width", default=128, help="Image width. Default is 128. Change depending on network resolution", type=int)
    parser.add_argument("--height", default=96, help="Image Height. default is 96. Change depending on network resolution", type=int)
    parser.add_argument("-d", "--depth", default=16, help="depth of the quadtree. Default is 16.", type=int)
    parser.add_argument("--input", help="Input directory. If not specified, will raise FileNotFoundError", type=str)
    parser.add_argument("--output", help="Output directory. Default is ../output", default=defaultdir, type=str)
    parser.add_argument("--file", help="Input Filename. Used to read input from input directory and create equivalent output", type=str)

    args = parser.parse_args()
    return args


if __name__=="__main__":
    dirname = os.path.dirname(os.path.abspath(__file__))

    # argument parsing
    defaultdir = os.path.join(dirname, '..', 'output')
    args = parse_args(defaultdir)

    # Quadtree values
    img_width = args.width
    img_height = args.height
    low = (args.lowx, args.lowy)
    scale =  args.scale
    max_depth = args.depth

    tree = qt.Quadtree(low=low, scale=scale, max_depth=max_depth)

    # symlinked - may have to use "realpath" or something
    bagf = os.path.join(args.input, args.file + ".bag")

    # rosbag setup
    pcl_topic = "/pcl_plane"
    ctr = 0

    # timing setup
    timerlist = []
    hz = 30
    
    try:
        with rosbag.Bag(bagf, 'r') as bag:
            duration = bag.get_end_time() - bag.get_start_time()

            for (topic, msg, ts) in tqdm(bag.read_messages(topics=pcl_topic), desc='progress', total=bag.get_message_count(topic_filters=pcl_topic)):

                # Decoding the points
                intensity_channel = msg.channels[1].values
                pts = msg.points
                val_dict = decode_intensity(pts, intensity_channel)
                
                # Inserting the points
                idcs = tree.insertion_idcs(pts, img_width, img_height)
                tree.insert_points_arr(values=val_dict.values(), idcs = idcs)

        # output saving
        a = datetime.now()
        d = a.strftime("%y-%m-%d_%H-%M")
        f = "{}_{}-qt-{}-{}-{}.pkl".format(d, args.file, max_depth, low, scale)
        outpath = os.path.join(args.output, f)
        tree.save(outpath)
    except FileNotFoundError: raise