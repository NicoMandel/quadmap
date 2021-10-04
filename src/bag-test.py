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

import quadtree as qt

def decode_intensity(pts, intensity):
        pt_dict = {}
        for i, ch in enumerate(intensity):
            pt = pts[i]
            pt_dict[tuple([pt.x, pt.y])] = int(ch)
        
        return pt_dict

def print_statistics(stop, start, timerlist, hz, duration):
    avg_time = np.asarray(timerlist).mean()
    print("Finished iteration {}, took: {}, average time: {}".format(ctr, stop-start, avg_time))

    completion_time = avg_time * hz * duration
    mts = completion_time // 60
    hrs = mts // 60
    mts = mts % 60
    s = completion_time % 60
    print("Assumed recording at {} hz, The bag of duration: {:1f}s will take {} hrs {} m {} s to complete".format(hz, duration,hrs, mts, s))



if __name__=="__main__":
    dirname = os.path.dirname(os.path.abspath(__file__))
    bagdir = os.path.join(dirname, '..', "rosbag")

    # Quadtree values
    # ! look up at which location the map is
    img_width = 128
    img_height = 96
    low = (-80, -80)
    scale =  120
    max_depth = 16 
    tree = qt.Quadtree(low=low, scale=scale, max_depth=max_depth)

    # symlinked - may have to use "realpath" or something
    bagf_name = "sim_tgt2-descend.bag"
    bagf = os.path.join(bagdir, bagf_name)

    # rosbag setup
    pcl_topic = "/pcl_plane"
    ctr = 0

    # timing setup
    timerlist = []
    hz = 30
    

    with rosbag.Bag(bagf, 'r') as bag:
        duration = bag.get_end_time() - bag.get_start_time()

        for (topic, msg, ts) in bag.read_messages(topics=pcl_topic):
            ctr += 1
            start = time.perf_counter()
            # quick test to just do the first 30 messages
            if ctr >= 15:
                break
            print("Starting iteration {}".format(ctr))

            intensity_channel = msg.channels[1].values
            pts = msg.points
            # val_dict = self.decode_cmap(pts, rgb_channel)
            val_dict = decode_intensity(pts, intensity_channel)
            # !idcs are always 1 - find out why, probably the type of the message
            idcs = tree.insertion_idcs(pts, img_width, img_height)
            # prs = self.tree.find_priors_arr(idcs)
            tree.insert_points_arr(values=val_dict.values(), idcs = idcs)
            stop = time.perf_counter()
            timerlist.append(stop-start)
            print_statistics(stop, start, timerlist, hz, duration)

    # output saving
    a = datetime.now()
    d = a.strftime("%y-%m-%d_%H-%M")
    f = "{}_{}-qt-{}-{}-{}.pkl".format(d, bagf_name, max_depth, low, scale)
    outpath = os.path.join(dirname, '..', 'output', f)
    tree.save(outpath)