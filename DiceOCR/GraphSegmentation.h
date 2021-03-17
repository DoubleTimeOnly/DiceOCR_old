#pragma once
#include "opencv2/core/core.hpp"
#include "DisjointSet.h"
#include <vector>
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
    void segmentGraph(const cv::Mat& image, float c=20000.0, int minsize=100);
    void calculateEdges(const cv::Mat& image);
    const int getNumEdges();
    cv::Mat drawSegments(bool drawboxes=true);
    void getROIs(std::vector<cv::Rect>& regions, int maxWidth=200, int maxHeight=200);
};
