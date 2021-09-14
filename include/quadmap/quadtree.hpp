#ifndef QUADTREE_H
#define QUADTREE_H

#include <math/math.hpp>
#include <unordered_map>

/**
 * @brief 
 * header file for the quadtree implementation used in this package. Based on the numerical recipes in C++ implementation
 * 
 */
class Quadtree
{
private:
    // Values that are fixed for the data structure
    static const int BIT = 4;
    static const int OFFSET = 2;
    static const int MAXDEPTH = 32 / 2;
    
    // constant values per tree
    const int maxd;
    double lo[2];
    double scale[2];

    // the unordered map storage container
    std::unordered_map<uint32_t, Quadtree> map;

    // functions
    void setbox();
    

public:
    Quadtree(/* args */);
    ~Quadtree();



};

#endif