#ifndef QUADTREE_H
#define QUADTREE_H

#include <math/math.hpp>
#include <unordered_map>
#include "pointbox.hpp"

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
    std::unordered_map<uint32_t, std::string> map;

    // Additional stuff
    int current_d;


public:
    Quadtree(/* args */);
    ~Quadtree();

    // Functions
    void setbox(Point low, Point high);
    Box getBox(uint32_t idx);
    uint32_t getIndex(Point pt);

    // Indexing functions
    uint32_t getDaughterNumber(uint32_t idx);
    uint32_t getMotherIdx(uint32_t idx);
    uint32_t getLeftDaughterIdx(uint32_t idx);
    uint32_t getRightDaughterIdx(uint32_t idx);

};

#endif