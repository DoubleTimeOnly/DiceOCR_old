#pragma once
#include <unordered_map>
#include "opencv2/core/core.hpp"

struct component
{
    int parent;
    int rank;
    int size;
    int minX = 0, minY = 0, maxX = 0, maxY = 0;
};

class DisjointSet
{
private:
    int rows, cols;
    int num_components;
    component* components = NULL;

public:
    // core functionality
    void makeset(cv::Mat const& image);
    int findRoot(int p);
    void mergeSets(int p1, int p2);

    // getters
    const cv::Rect getBoundingBoxCoordinates(int root);
    const int findRank(int p) { return this->components[findRoot(p)].rank; }
    const int getNumComponents() { return this->num_components; }
    const int getSetSize(int p) { return this->components[p].size; }
};



