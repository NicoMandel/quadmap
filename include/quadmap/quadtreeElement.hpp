#ifndef QUADTREEELEMENT_H
#define QUADTREEELEMENT_H

#include <cstdint>

class QuadtreeElement{
private:
    double probability;
    uint32_t index;
public:
    QuadtreeElement(uint32_t idx, double prob = -1.0) : probability(prob), index(idx) {};
    ~QuadtreeElement() {};
    bool insert(double probability);
    bool update(double probability);
    uint8_t getLevel();
};

#endif