#!/usr/bin/env python

import cv_bridge
import rospy
from quadmap.srv import getMap, getMapResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import quadtree as qt
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class QTclient:

    def __init__(self) -> None:
        rospy.wait_for_service("getMap")
        self.bridge = cv_bridge.CvBridge()
        try:
            self.getm = rospy.ServiceProxy("getMap", getMap)
        except rospy.ServiceException as se:
            rospy.logerr("could not create client: {}".format(se))

        max_depth = rospy.get_param("max_depth", default=14)
        scale = rospy.get_param("qt_scale", default=50)
        self.tree = qt.Quadtree(scale=scale, max_depth=max_depth)

        self.pub = rospy.Publisher("qtImage", Image, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(1.), self.requestMap)

    def requestMap(self, timer) -> None:
        # request the map location
        try:
            resp = self.getm(1)
        except rospy.ServiceException as se:
            rospy.logerr("could not request quadtree: {}".format(se))

        fpath = resp.path
        # read the map from file
        treedict = self.readMap(fpath)
        self.tree.dictionary = treedict
        
        # convert the map to an image
        img = self.convertToImg()
        try:
            img_msg = self.bridge.cv2_to_imgmsg(img)
            self.pub.publish(img_msg)

        except CvBridgeError as cve:
            rospy.logerr("Cv Bridge Error: {}".format(cve))

    def readMap(self, fname):
        """
            Function to read the map from a filename
        """
        with open(fname, 'r') as json_data:
            data = json.load(json_data)
        return data

    def convertToImg(self):
        """
            Follow this example: https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        """
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        self.tree.plot_tree(ax)
        ax.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0)
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image




if __name__=="__main__":
    qtc = QTclient()
    nodename = type(qtc).__name__.lower()
    rospy.init_node(nodename)

    try:
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr_once(e)