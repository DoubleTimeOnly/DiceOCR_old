#pragma once
#include "opencv2/core/core.hpp"
struct edge
{
    std::pair<int, int> a, b;
    float weight;
};
class GraphSegmentation
{
private:
    edge* edges;
    void getEdges(const cv::Mat& image);
    int num_edges = 0;

public:
    void segmentGraph(const cv::Mat& image);
    edge* calculateEdges(const cv::Mat& image);
    const int getNumEdges();
};