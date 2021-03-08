#pragma once
#include "opencv2/core/core.hpp"
#include "DisjointSet.h"
struct edge
{
    int a, b;
    uchar weight;
};

class GraphSegmentation
{
private:
    edge* edges = NULL;
    int rows, cols;
    int num_edges;
    DisjointSet components;


public:
    void segmentGraph(const cv::Mat& image, float c, int minsize);
    void calculateEdges(const cv::Mat& image);
    const int getNumEdges();
    cv::Mat drawSegments();
};
