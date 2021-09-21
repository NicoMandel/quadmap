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

    uint32_t node=1, left, right, children;
    uint8_t level;
    for (level = 2; level <= this->maxd; level++){
        left = this->getLeftDaughterIdx(node);
        right = this->getRightDaughterIdx(node);
        for (children=left; children <= right; children++){
            if (pt.isinbox(getBox(children))){
                node = children; break;
            }
        }
        // Not sure what this safety is for? Integer overflow? Or if it hasn't been found not to be cyclic?
        if (children > right) break;
    }
    return node;
}

// Getting the insertion indices in a smarter way
std::unordered_map<uint32_t, Point> Quadtree::getInsertion(std::vector<Point> points){
    uint8_t level;

    uint32_t left, right;

    std::unordered_map<uint32_t, Point> insertions;
    // using containers to keep what we need - the 
    std::vector<uint32_t> idcs_vec(points.size(), 1);
    std::unordered_map<uint32_t, uint32_t> idcs_map;
    
    // step through the levels
    for (level = 2; level<= this->maxd; level++)
    {
        // point has the function that tests whether it is in a box
        for (int i = 0; i < points.size(); i ++)
        {
            Point pt = points.at(i);
            uint32_t pt_idx = idcs_vec.at(i);
            left = getLeftDaughterIdx(pt_idx);
            right = getRightDaughterIdx(pt_idx);
            for (int j = left; j <= right; j++){
                if (pt.isinbox(getBox(j))){
                    idcs_vec.at(i) = j;
                    idcs_map[i] = j;
                    break;
                }
            // Safety?
            if (j > right) break;
            }
        }

        // Now do the check. Every point now has a an index for the level that we are at
        for (int i = 0; i< points.size(); i++){
            std::vector<uint32_t> neighbors = getNeighborsIdcs(idcs_vec.at(i));
            // TODO this is O(n^2 log n)... fucking useless!
        }

    }

    return insertions;
}

// Get the indices of multiple points
std::unordered_map<uint32_t, Point> Quadtree::getIndices(std::vector<Point> pts){
    std::unordered_map<uint32_t, Point> outputs;
    
    // May have to move this allocation over for OpenMP to work
    uint32_t idx;
    for (int i = 0; i<pts.size(); i++){
        idx = getIndex(pts.at(i));
        outputs[idx] = pts.at(i);
    }
    return outputs;
}


// TODO: could use Eigen here...
std::vector<uint32_t> Quadtree::getIndicesVec(std::vector<Point> pts, int width, int height){
    // for each point, have a vector with the length of depth
    std::vector<std::vector<uint32_t>> idcs(pts.size(), std::vector<uint32_t>(maxd));
    std::vector<bool> found(pts.size(), false);

    uint8_t level;
    // for every level, iterate over all the points
    for (level = 2; level <= maxd; level++){
        // for every point
        for (int i = 0; i < pts.size(); i++){
            uint32_t curr_idx = idcs.at(i).at(level-1);
            // if the index has not been found yet
            if (!found.at(i)){
                Point curr_pt = pts.at(i);
                // find the child index
                uint32_t ch_idx = getChild(curr_idx, curr_pt);
                // insert the child index into the vector of vectors
                idcs.at(i).at(level) = ch_idx;
            }
            else{
                idcs.at(i).at(level) = curr_idx;
            }
        }

        // Now iterate over the points again - and look at the neighborhood
        for (int i = 0; i < pts.size(); i++){
            if (!found.at(i)){
                uint32_t own_idx = idcs.at(i).at(level);
                std::vector<uint32_t> neighbors = getNeighborsIdcs(i, width, height);
                // set the default value of "found" to true
                bool f = true;
                // for every neighbor
                for (int j = 0; j < neighbors.size(); j++){
                    // we are still in "level" - look at the index the neighbor has 
                    uint32_t neighbor_idx = idcs.at(neighbors.at(j)).at(level);
                    // if any of the neighbors has the same index
                    if (neighbor_idx == own_idx) f = false;
                }
                found.at(i) = f;
            }
        }
    }

    for (int i = 0; i<pts.size(); i++){
        idcs.at(i) = getIndex(pts.at(i));
    }
    return idcs.at(level);
}

// get the index of the child -> this will always work, since a point has 0 extent - so it will always be there
uint32_t Quadtree::getChild(uint32_t m_idx, Point pt){
    uint32_t left, right;
    left = getLeftDaughterIdx(m_idx);
    right = getRightDaughterIdx(m_idx);
    for (uint32_t child = left; child <= right; child++){
        if (pt.isinbox(getBox(child))) return child;
    }
}

// The index reducing function. Is currently O(n^2)
std::unordered_map<uint32_t, Point> Quadtree::reduceIdcs(std::unordered_map<uint32_t, Point> &idcs){
    // from: https://www.techiedelight.com/convert-map-vector-key-value-pairs-cpp/
    std::vector<uint32_t> frontier(idcs.size());
    std::unordered_map<uint32_t, Point> reduced_idcs = idcs;
    std::copy(idcs.begin(), idcs.end(), frontier.begin());       

    // Now look through the search frontier
    uint32_t curr_idx, mother;
    // std::vector<uint32_t> mothers(frontier.size());
    while(!frontier.empty()){
        // get all the mothers
        for (int i=0; i<frontier.size(); i++)
        {
            // Brute force approach - there is definitely a smarter way to do this
            curr_idx = frontier.at(0);          // get the first element
            frontier.erase(frontier.begin());   // remove it from the list

            mother = getMotherIdx(curr_idx);

            // for every other element in the frontier 
            for (int i=0; i<frontier.size(); i++)           // may have to move this to i=1
            {
                if (isSibling(curr_idx, frontier.at(i)))
                {
                    reduced_idcs[curr_idx] = idcs[curr_idx];
                    reduced_idcs[frontier.at(i)] = idcs[frontier.at(i)];
                    // remove the sibling from the frontier
                    frontier.erase(frontier.begin() + i);
                    // initialising the point to negative
                    // ! this knowledge should be used later - can be implemented as a check
                    // if the mother does not yet exist
                    // if ()
                    reduced_idcs[mother] = Point(-1., -1.);
                }
            }
            // push the mother in the back. If they are siblings, they have the same mother, so this only needs to be added once
            // If they are not siblings the mother has to be pushed back anyways.
            frontier.push_back(mother);
            frontier.shrink_to_fit();
        }
    }
}

// Function to reduce the idcs on the idcs of the points alone
std::vector<uint32_t> reduceIdcs(std::vector<uint32_t> pt_idcs, int width, int height){
    std::vector<uint32_t> reduced_idcs(pt_idcs.size());
    std::copy(pt_idcs.begin(), pt_idcs.end(), reduced_idcs.begin());

    std::vector<uint32_t> frontier(pt_idcs.size());
    std::copy(pt_idcs.begin(), pt_idcs.end(), frontier.begin());

    uint32_t curr_idx;
    std::vector<uint32_t> neighbors;

    // for frontier.empty() to work, we need a shrink_to_fit() in the end
    while(!frontier.empty()){
        curr_idx = frontier.at(0);          // get the first element
        frontier.erase(frontier.begin());   // remove it from the list
        
        // ! This does not work
        neighbors = getNeighborsIdcs(pt_idcs.at(curr_idx), width, height);


        frontier.shrink_to_fit();
    }       



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

// Boolean functions - with ternary operator
bool Quadtree::isSibling(uint32_t idx, uint32_t tst_idx){
    return (getMotherIdx(idx) == getMotherIdx(tst_idx)) ? true :  false;
}

// function to find the vector of the Moore neighborhood - already does row and column checks, so length is not consistent
std::vector<uint32_t> getNeighborsIdcs(uint32_t idx, int width, int height){
    std::vector<uint32_t> neighbors;
    int col = idx % width;
    int row = std::floor(idx / width);
    // from -1 to +1
    for (int i = -1; i <= 1; i++)       // for every row
    {
        for (int j = -1; j <= 1; j++)       // for every column
        {
            // TODO - add col and row in here
            uint32_t r = row + i;
            uint32_t c = col + j;
            if(c >= 0 && c < width && r >= 0 && r < height) neighbors.push_back(r * width + c);
        }
    }

    return neighbors;
}

