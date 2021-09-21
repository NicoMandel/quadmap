#ifndef QUADTREE_H
#define QUADTREE_H

#include <math/math.hpp>
#include <unordered_map>
#include "pointbox.hpp"
#include "quadtreeElement.hpp"
#include <unordered_set>

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
    int maxd;
    double lo[2];
    double scale[2];

    // the unordered map storage container
    // TODO: for now, these are just doubles
    std::unordered_map<uint32_t, double> map;

    // Additional stuff
    int current_d;

    // Private member functions
    bool setMothers(uint32_t idx);

    std::vector<uint32_t> getNeighborsIdcs(uint32_t idx, int width, int height);
    uint32_t getChild(uint32_t m_idx, Point pt);
   

public:
    Quadtree(/* args */);
    Quadtree(int);
    ~Quadtree();

    // Functions for singular use
    void setbox(Point low, Point high);
    Box getBox(uint32_t idx);
    uint32_t getIndex(Point pt);

    // Indexing functions
    uint32_t getDaughterNumber(uint32_t idx);
    uint32_t getMotherIdx(uint32_t idx);
    uint32_t getLeftDaughterIdx(uint32_t idx);
    uint32_t getRightDaughterIdx(uint32_t idx);

    // Functions using containers for multiple elements
    // TODO: turn this into a hash map
    std::unordered_map<uint32_t, Point> getIndices(std::vector<Point> pts);
    std::vector<uint32_t> Quadtree::getIndicesVec(std::vector<Point> pts, int width);         // the non-hash map version, where the indices are corresponding 
    std::unordered_map<uint32_t, Point> reduceIdcs(std::unordered_map<uint32_t, Point>&);
    std::vector<uint32_t> reduceIdcs(std::vector<uint32_t> pt_idcs, int width, int height);

    std::unordered_map<uint32_t, Point> Quadtree::getInsertion(std::vector<Point> points);

    // test function
    bool isSibling(uint32_t idx, uint32_t tst_idx);


};

#endif