#include "quadmap/quadtree.hpp"

/**
 * @brief 
 * C++ implementation of the Quadtree already written in python
 * All includes should be done in the .hpp file included at the top
 */


Quadtree::Quadtree(): maxd(maxd) {
    if (maxd > this->MAXDEPTH) throw std::runtime_error(std::string("Exceeding maximum depth: %d larger than %d", maxd, this->MAXDEPTH));
    setbox(Point(0.0, 0.0), Point(1.0, 1.0));

}

Quadtree::~Quadtree(){

}

void Quadtree::setbox(Point low, Point high){
    this->lo[0] = low.get(0);
    this->lo[1] = low.get(1);
    
    this->scale[0] = high.get(0) - low.get(0);
    this->scale[1] = high.get(1) - low.get(1);
}

// Get the box surrounding an index
Box Quadtree::getBox(uint32_t idx){
    // uint32_t j;
    uint8_t i, j;
    Point low, high;
    double offset[2] = {0.0, 0.0};
    double delta = 1.0;

    // traverse back up through the tree and add the offset
    while (idx > 1){
        j = getDaughterNumber(idx);
        // the daughter number is always between 0 and 3 - the 2 LSBs dictate in which quadrant the element sits.
        if (j & 1) offset[0] += delta;
        if (j & (1 << 1)) offset[1] += delta;       // 1 << 1 evaluates to 2... 

        idx = getMotherIdx(idx);
        delta *= 2.0;       // double the delta - because the second box is twice as big
    }
    // Scale the offsets
    double l, h;
    l = lo[0] + scale[0] * offset[0] / delta;
    low.setx(l);
    l = lo[1] + scale[1] * offset[1] / delta;
    low.sety(l);
    h = lo[0] + scale[0] * (offset[0] + 1.0) / delta;
    high.setx(h);
    h = lo[1] + scale[1] * (offset[1] + 1.0) / delta;
    high.sety(h);

    return Box(low, high);
}

// Get the lowest index where point pt should be inserted
uint32_t Quadtree::getIndex(Point pt){
    // ! CONTINUE HERE
    // TODO ====> continue with the function that calculates the lowest level index
    // ! CONTINUE HERE
}

// Indexing functions
// Getting the number of the daughter
uint32_t Quadtree::getDaughterNumber(uint32_t idx){
    return (idx + this->OFFSET) % this->BIT;
}


// Get the index of the mother
uint32_t Quadtree::getMotherIdx(uint32_t idx){
    return (idx + this->OFFSET) >> 2;
}

// Getting the index of the leftmost daughter
uint32_t Quadtree::getLeftDaughterIdx(uint32_t idx){
    return (BIT * idx) - OFFSET;
}

// Getting the index of the rightmost daughter
uint32_t Quadtree::getRightDaughterIdx(uint32_t idx){
    return (getLeftDaughterIdx(idx) + BIT - 1); 
}
