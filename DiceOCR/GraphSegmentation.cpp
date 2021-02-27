#include "opencv2/core/core.hpp"
#include "GraphSegmentation.h"
#include "DisjointSet.h"

float calculateEdgeWeight(const cv::Mat& image, const std::pair<int, int>& p1, const std::pair<int, int>& p2)
{
    // Mat.at is (y,x)
    uchar intensity1 = image.at<uchar>(p1.second, p1.first);
    uchar intensity2 = image.at<uchar>(p2.second, p2.first);
    float weight = std::abs(intensity1 - intensity2);
    return weight;
}

edge* GraphSegmentation::calculateEdges(const cv::Mat& image)
{
    const int rows = image.rows;
    const int cols = image.cols;
    edges = new edge[rows*cols*4];    // create array of edges

    num_edges = 0;
    cv::Mat component_weights = cv::Mat::zeros(rows, cols, CV_32F);
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            float min_weight = 100000;
            std::pair<int, int> p1(col, row);
            // check vertical edges
            if (row < rows - 1)
            {
                edges[num_edges].a = p1;
                std::pair<int, int> p2(col, row + 1);
                edges[num_edges].b = p2;
                float weight = calculateEdgeWeight(image, p1, p2);
                edges[num_edges].weight = weight;

                if (weight < min_weight) { min_weight = weight; }
                num_edges++;
            }
            // check horizontal edges
            if (col < cols - 1)
            {
                edges[num_edges].a = p1;
                std::pair<int, int> p2(col+1, row);
                edges[num_edges].b = p2;
                float weight = calculateEdgeWeight(image, p1, p2);
                edges[num_edges].weight = weight;

                if (weight < min_weight) { min_weight = weight; }
                num_edges++;
            }
            // check south-east edges
            if ((row < rows - 1) && (col < cols - 1))
            {
                edges[num_edges].a = p1;
                std::pair<int, int> p2(col+1, row+1);
                edges[num_edges].b = p2;
                float weight = calculateEdgeWeight(image, p1, p2);
                edges[num_edges].weight = weight;

                if (weight < min_weight) { min_weight = weight; }
                num_edges++;
            }
            // check north-east edges
            if ((row > 0) && (col < cols - 1))
            {
                edges[num_edges].a = p1;
                std::pair<int, int> p2(col+1, row-1);
                edges[num_edges].b = p2;
                float weight = calculateEdgeWeight(image, p1, p2);
                edges[num_edges].weight = weight;

                if (weight < min_weight) { min_weight = weight; }
                num_edges++;
            }
            component_weights.at<float>(row, col) = min_weight;
        }
    }
    return edges;

}

void GraphSegmentation::segmentGraph(const cv::Mat& image)
{
    calculateEdges(image);
    std::sort(edges, edges + num_edges);
}
const int GraphSegmentation::getNumEdges()
{
    return num_edges;
}