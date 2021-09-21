#include "quadmap/quadtree.hpp"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <geometry_msgs/Point32.h>


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
public:
    QuadMap_node(/* args */);
    ~QuadMap_node();
    
    // callback function
    void pcl_callback(sensor_msgs::PointCloudConstPtr);
    bool getQuadmap();
};

// constructor
QuadMap_node::QuadMap_node(/* args */) : nh_(nh_)
{
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
void QuadMap_node::pcl_callback(sensor_msgs::PointCloudConstPtr pcl){
    // pass it to the quadtree
}

// Service function
bool QuadMap_node::getQuadmap(){};


int main(int argc, char** argv){
    ros::init(argc, argv, "quadmap_node");
    QuadMap_node map_node();
    ros::spin();
    return 0;
}