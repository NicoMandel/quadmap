#!/usr/bin/env python

import os.path
import rosbag
from datetime import datetime

import quadtree as qt

def decode_intensity(pts, intensity):
        pt_dict = {}
        for i, ch in enumerate(intensity):
            pt = pts[i]
            pt_dict[tuple([pt.x, pt.y])] = int(ch)
        
        return pt_dict

if __name__=="__main__":
    dirname = os.path.dirname(os.path.abspath(__file__))
    bagdir = os.path.join(dirname, '..', "rosbag")

    # Quadtree values
    # ! look up at which location the map is
    img_width = 128
    img_height = 96
    low = (-80, 40)
    scale =  100
    max_depth = 16 
    tree = qt.Quadtree(low=low, scale=scale, max_depth=max_depth)

    # symlinked - may have to use "realpath" or something
    bagf_name = "sim_tgt2-descend.bag"
    bagf = os.path.join(bagdir, bagf_name)

    # rosbag setup
    pcl_topic = "/pcl_plane"
    ctr = 0
    with rosbag.Bag(bagf, 'r') as bag:
        for (topic, msg, ts) in bag.read_messages(topics=pcl_topic):
            ctr += 1
            # quick test to just do the first 30 messages
            if ctr >= 30:
                break
            intensity_channel = msg.channels[1].values
            pts = msg.points
            # val_dict = self.decode_cmap(pts, rgb_channel)
            val_dict = decode_intensity(pts, intensity_channel)
            idcs = tree.insertion_idcs(pts, img_width, img_height)
            # prs = self.tree.find_priors_arr(idcs)
            tree.insert_points_arr(values=val_dict.values(), idcs = idcs)

    # output saving
    a = datetime.now()
    d = a.strftime("%y-%m-%d_%H-%M")
    f = "{}_{}-qt-{}-{}-{}.pkl".format(d, bagf_name, max_depth, low, scale)
    outpath = os.path.join(dirname, '..', 'output', f)
    tree.save(outpath)