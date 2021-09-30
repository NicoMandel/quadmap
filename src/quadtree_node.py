#!/usr/bin/env python

import struct

import rospy
from sensor_msgs.msg import PointCloud
# import quadtree as qt
import quadtree as qt
import numpy as np

# for storing / sending the quadmap
# from quadmap.srv import getMap, getMapResponse
# import json
import os.path

# For plotting the quadmap
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
# from cv_bridge import CvBridge, CvBridgeError
# from sensor_msgs.msg import Image

from datetime import datetime

class QuadMap_Node:

    def __init__(self) -> None:
        topic = rospy.get_param("pcl_topic", default="pcl_plane")
        max_depth = rospy.get_param("max_depth", default=16)
        scale = rospy.get_param("qt_scale", default=70)
        self.experiment = rospy.get_param("experiment", default="default-exp")
        low = rospy.get_param("low", default=-40)

        self.img_width = rospy.get_param("out_width", default=256)
        self.img_height = rospy.get_param("out_height", default=192)
        
        # initialise the tree
        # TODO: set these bounds accordingly 
        self.tree = qt.Quadtree(low=low, scale=scale, max_depth=max_depth)
        
        # advertising a service
        # self.serv = rospy.Service('getMap', getMap, self.getMap)

        # file directory to write to 
        directory = "~/.ros"
        fname = "quadtree.json"
        self.fpath = os.path.abspath(os.path.expanduser(os.path.join(directory, fname)))
        # creating the file without doing anything
        open(self.fpath, 'a').close()     

        # subscriber to the pcl_plane
        rospy.Subscriber(topic, PointCloud, self.pcl_callback, queue_size=1)

        rospy.on_shutdown(self.shutdown)

        # Add a publisher for an image topic.
        # px = 1/plt.rcParams['figure.dpi']
        # self.bridge = CvBridge()
        # self.fig = Figure(figsize=(1400*px,1400*px))
        # self.canvas = FigureCanvas(self.fig)
        # self.ax =self.fig.gca()

        # self.img_pub = rospy.Publisher("qt_img", Image, queue_size=1)



    def pcl_callback(self, msg):
        # rgb_channel = msg.channels[0].values
        intensity_channel = msg.channels[1].values
        pts = msg.points
        # val_dict = self.decode_cmap(pts, rgb_channel)
        val_dict = self.decode_intensity(pts, intensity_channel)
        idcs = self.tree.insertion_idcs(pts, self.img_width, self.img_height)
        # prs = self.tree.find_priors_arr(idcs)
        self.tree.insert_points_arr(values=val_dict.values(), idcs = idcs)

        # TODO: after inserting points into the normal quadtree - what to do next? 
        # ! pruning does not make sense - only in post - because otherwise the values would get inserted multiple times

        # reduced_idcs_dict = self.tree.reduce_idcs(idcs)
        # TODO: figure out which one of these is 
        # self.tree.insert_points(reduced_idcs_dict)

        # sending out on an image topic
        # self.send_img()


    def send_img(self):
        # self.tree.plot_tree(self.ax)
        # # self.ax.axis('off')
        # # self.fig.tight_layout(pad=0)
        # # self.ax.margins(0)
        # self.canvas.draw()
        # img = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8')
        # img = img.reshape(self.fig.canvas.get_width_height()[::-1]+ (3,))
        # try:
        #     imgmsg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
        # except CvBridgeError as cve:
        #     rospy.logerr("Encountered Error converting image message: {}".format(
        #         cve
        #     ))
        # self.img_pub.publish(imgmsg)
        pass
    
    def decode_intensity(self, pts, intensity):
        pt_dict = {}
        for i, ch in enumerate(intensity):
            pt = pts[i]
            pt_dict[tuple([pt.x, pt.y])] = int(ch)
        
        return pt_dict
    
    def decode_cmap(self, pts, col_channel):
        pt_dict = {}
        for i, ch in enumerate(col_channel):
            r, g, b = self.decode_value(ch)

            pt = pts[i]
            pt_dict[tuple([pt.x, pt.y])] = tuple([r, g, b])
        
        return pt_dict


    def decode_value(self, ch):
        """
            Looky here: https://answers.ros.org/question/340159/rgb-color-data-in-channelfloat32-of-pointcloud-message-not-working/
            https://answers.ros.org/question/35655/getting-pointcloud2-data-in-python/
            struct.unpack('f', struct.pack('i', 0xff0000))[0]
        """
        float_rgb = struct.unpack('f', struct.pack('i', ch))[0]
        r = (float_rgb[0] & (255 << 16))
        g = (float_rgb[0] & (255 << 8))
        b = (float_rgb[0] & (255))

        return r, g, b

    def getMap(self, req):
        """
            Function to respond to the Map request
        """
        pass 
        # with open(self.fpath, 'w') as fp:
        #     json.dump(self.tree.dictionary, fp)
        # rospy.loginfo("Wrote Quadtree dictionary to: {}".format(self.fpath))
        # return getMapResponse(self.fpath)

    def shutdown(self):
        """
            shutdown function to save the tree
        """
        now = datetime.now()
        d = now.strftime("%y-%m-%d_%H-%M")
        outputdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
        f = "{}_{}-qt-{}-{}-{}.pkl".format(self.experiment, d, self.tree.max_depth, self.tree.low, self.tree.scale)
        fnam = os.path.join(outputdir, f)
        self.tree.save(fnam)
        rospy.logwarn("Saved quadtree to: {}".format(fnam))

        


if __name__=="__main__":
    qmn = QuadMap_Node()
    nodename = type(qmn).__name__.lower()
    rospy.init_node(nodename)
    # direct = os.path.dirname(os.path.abspath(__file__)) 

    # fig = plt.figure(figsize=(15,9))
    # ax = fig.gca()

    # def animate(i):
    #     print(i)
    #     ax.clear()
    #     qmn.tree.plot_tree(ax)
    #     if (i == 4):
    #         fname = os.path.join(direct, 'test_tree_{}s.pkl'.format(i*2))
    #         qmn.tree.save(fname)
    
    # ani = matplotlib.animation.FuncAnimation(fig, animate, interval=2000)
    # plt.show()

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException as e:
            rospy.logerr_once(e)