#include "quadmap/quadtree.hpp"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>
#include <sensor_msgs/ChannelFloat32.h>


// const ptrs in C++ https://answers.ros.org/question/212857/what-is-constptr/

/**
 * @brief ROS node encapsulating the C++ implementation of the quadmap. Exposes Services and topics
 * 
 */

class QuadMap_node
{
private:
    ros::NodeHandle nh_;
    ros::Subscriber pcl_sub_;
    Quadtree qt;
    Point low, high;

public:
    QuadMap_node(/* args */);
    ~QuadMap_node();
    
    // callback function
    void pcl_callback(sensor_msgs::PointCloudConstPtr&);
    bool getQuadmap();
};

// constructor
QuadMap_node::QuadMap_node(/* args */) : nh_(nh_)
{
    // Parameters to initialise the QuadTree
    int scale, max_depth;
    double lp;
    nh_.param("scale", scale, 50);
    nh_.param("max_depth", max_depth, 16);
    nh_.param("low", lp, 0.0);

    this->qt = Quadtree(max_depth);

    // these are the default points that set the map
    low = Point(lp, lp);
    high = Point(low.getx() + scale, low.gety() + scale);
    qt.setbox(low, high);

    ROS_INFO("Initialised a Quadtree to store information. Depth is %d, low is: (%f, %f), high: (%f, %f)",
        max_depth, low.getx(), low.gety(), high.getx(), high.gety()
    );



    std::string pcl_topic;
    nh_.getParam("pcl_topic", pcl_topic);
    nh_.subscribe(pcl_topic, 1, &QuadMap_node::pcl_callback, this);
    
    // Advertise a service to get the map
    // ros::ServiceServer service = nh_.advertiseService("getQuadmap", QuadMap_node::getQuadmap, this);
}

// destructor
QuadMap_node::~QuadMap_node()
{
}


// Callback
void QuadMap_node::pcl_callback(sensor_msgs::PointCloudConstPtr& pcl){
    std::vector<geometry_msgs::Point32> pts = pcl->points;
    std::vector<float> intensity = pcl->channels.at(1).values;

    // insert the points into the quadtree

    // FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK 
}

// Service function
bool QuadMap_node::getQuadmap(){}


int main(int argc, char** argv){
    ros::init(argc, argv, "quadmap_node");
    QuadMap_node map_node();
    ros::spin();
    return 0;
}