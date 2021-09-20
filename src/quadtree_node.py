#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud
import quadtree as qt
import numpy as np

class QuadMap_Node:

    def __init__(self) -> None:
        topic = rospy.get_param("pcl_topic", default="pcl_plane")
        max_depth = rospy.get_param("max_depth", default=14)
        scale = rospy.get_param("qt_scale", default=50)
        
        # initialise the tree
        self.tree = qt.Quadtree(low=(0, 0), scale=scale, max_depth=max_depth)
        
        # advertising a service
        # TODO: make this a service that writes to a "tmp" directory and passes the filename

        # subscriber to the pcl_plane
        rospy.Subscriber(topic, PointCloud, self.pcl_callback, queue_size=1)


    def pcl_callback(self, msg):
        rgb_channel = msg.channels.values
        pts = msg.points
        val_dict = self.decode_cmap(pts, rgb_channel)
        idcs = self.tree.find_idcs(val_dict)

        reduced_idcs_dict = self.tree.reduce_idcs(idcs)
        self.tree.insert_points(reduced_idcs_dict)

    
    
    def decode_cmap(self, pts, channel):
        pt_dict = {}
        for i, ch in enumerate(channel):
            r, g, b = self.decode_value(ch)

            pt = pts[i]
            pt_dict[tuple([pt.x, pt.y])] = tuple([r, g, b])
        
        return pt_dict


    def decode_value(self, ch):
        r = (ch & (255 << 16))
        g = (ch & (255 << 8))
        b = (ch & (255))

        return r, g, b



if __name__=="__main__":
    qmn = QuadMap_Node()
    nodename = type(qmn).__name__.lower()
    rospy.init_node(nodename)

    try:
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr_once(e)