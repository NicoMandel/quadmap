#!/usr/bin/env python

import struct
import rospy
from sensor_msgs.msg import PointCloud
import quadtree as qt
import numpy as np
from quadmap.srv import getMap, getMapResponse
import json

class QuadMap_Node:

    def __init__(self) -> None:
        topic = rospy.get_param("pcl_topic", default="pcl_plane")
        max_depth = rospy.get_param("max_depth", default=14)
        scale = rospy.get_param("qt_scale", default=50)
        
        # initialise the tree
        self.tree = qt.Quadtree(low=(0, 0), scale=scale, max_depth=max_depth)
        
        # advertising a service
        # TODO: make this a service that writes to a "tmp" directory and passes the filename
        self.serv = rospy.Service('getMap', getMap, self.getMap)

        # file directory to write to 
        self.directory = "~/.ros/"
        self.fname = "quadtree.json"

        # subscriber to the pcl_plane
        rospy.Subscriber(topic, PointCloud, self.pcl_callback, queue_size=1)


    def pcl_callback(self, msg):
        rgb_channel = msg.channels[0].values
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
        nch = struct.unpack("f", ch)
        r = (nch[0] & (255 << 16))
        g = (nch[0] & (255 << 8))
        b = (nch[0] & (255))

        return r, g, b

    def getMap(self, req):
        """
            Function to respond to the Map request
        """
        fstring = self.directory + self.fname
        with open(fstring, 'w') as fp:
            json.dump(self.tree.dictionary, fp)
        rospy.loginfo("Wrote Quadtree dictionary to: {}".format(fstring))
        return getMapResponse(fstring)



if __name__=="__main__":
    qmn = QuadMap_Node()
    nodename = type(qmn).__name__.lower()
    rospy.init_node(nodename)

    try:
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr_once(e)