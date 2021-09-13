#ifndef QUADTREE_H
#define QUADTREE_H

#include <stdint.h>
#include <math/math.hpp>
/**
 * @brief 
 * header file for the quadtree implementation used in this package. Based on the numerical recipes in C++ implementation
 * 
 */
class Quadtree
{
private:
    static const int BIT = 4;
    static const int OFFSET = 2;
    // static in MAX_DEPTH = pow((double)2, (double)32);
public:
    Quadtree(/* args */);
    ~Quadtree();
};

#endif