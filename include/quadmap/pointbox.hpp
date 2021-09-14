#ifndef POINTBOX_H
#define POINTBOX_H

/**
 * @brief Point and Box classes for the implementation of the Quadtree 
 * TODO: require extension to work with the pointclouds sent out by ROS
 */

class Point{
private:
    double x[2];
public:
    Point(const Point &p){
        this->x[0] = p.get(0);
        this->x[1] = p.get(1);
    }

    Point(double x = 0.0, double y = 0.0){
        this->x[0] = x;
        this->x[1] = y;
    }

    // constructor for assignment
    Point& operator=(const Point &p){
        this->x[0] = p.get(0);
        this->x[1] = p.get(1);
        return *this;
    }
    
    ~Point(){
        // Correct??
        delete x;
    }

    double get(const int i) const{
        return x[i];
    }

    double getx(const int i) const{
        return x[0];
    }

    double gety(const int i) const{
        return x[1];
    }

    void set(double x, double y){
        this->x[0] = x;
        this->x[1] = y;
    }

    void sety(double y){
        this->x[1] = y;
    }

    void setx(double x){
        this->x[0] = x;
    }
    
    bool operator== (const Point &p) const {
        if (this->x[0] == p.get(0) && this->x[1] == p.get(1)) return true;
        else return false;
    }

    // to test whether a point is in a box
    bool isinbox(const Box &box){
        if (this->x[0] >= box.low().get(0) && this->x[0] <= box.high().get(0) &&
            this->x[1] >= box.low().get(1) && this->x[1] <= box.high().get(1)) return true;
        else return false;
    }
};

class Box{
private:
    Point lo;
    Point hi;
public:
    Box(const Point &lo, const Point &hi): lo(lo), hi(hi) {}
    ~Box(){}
    Point low() const { return this->lo;}
    Point high() const { return this->hi;}
};


// Distance calculations omitted


#endif