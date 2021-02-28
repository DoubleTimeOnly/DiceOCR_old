#pragma once
#include "opencv2/core/core.hpp"
#include "DisjointSet.h"
struct edge
{
    int a, b;
    float weight;
};

bool operator<(const edge& a, const edge& b) { return a.weight < b.weight; }

class GraphSegmentation
{
private:
    edge* edges;
    int rows, cols;
    int num_edges = 0;
    DisjointSet components;


public:
    void segmentGraph(const cv::Mat& image, float c, int minsize);
    void calculateEdges(const cv::Mat& image);
    const int getNumEdges();
    cv::Mat drawSegments();
};
