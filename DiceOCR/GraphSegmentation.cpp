#include "opencv2/core/core.hpp"
#include "GraphSegmentation.h"
#include "DisjointSet.h"

float calculateEdgeWeight(const cv::Mat& image, int row1, int col1, int row2, int col2)
{
    // Mat.at is (y,x)
    int intensity1 = image.at<uchar>(row1, col1);
    int intensity2 = image.at<uchar>(row2, col2);
    float weight = std::abs(intensity1 - intensity2);
    return weight;
}

int flattenedIdx(int row, int col, int max_cols)
{
    return row * max_cols + col;
}

void GraphSegmentation::calculateEdges(const cv::Mat& image)
{
    rows = image.rows;
    cols = image.cols;
    edges = new edge[rows*cols*4];    // create array of edges

    num_edges = 0;
    int p1;
    float weight;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            float max_weight = 0.0;
            p1 = flattenedIdx(row, col, cols);
            // check vertical edges
            if (row < rows - 1)
            {
                edges[num_edges].a = p1;
                edges[num_edges].b = flattenedIdx(row + 1, col, cols);
                weight = calculateEdgeWeight(image, row, col, row + 1, col);
                edges[num_edges].weight = weight;
                num_edges++;
            }
            // check horizontal edges
            if (col < cols - 1)
            {
                edges[num_edges].a = p1;
                edges[num_edges].b = flattenedIdx(row , col + 1, cols);
                weight = calculateEdgeWeight(image, row, col, row, col + 1);
                edges[num_edges].weight = weight;
                num_edges++;
            }
            // check south-east edges
            if ((row < rows - 1) && (col < cols - 1))
            {
                edges[num_edges].a = p1;
                edges[num_edges].b = flattenedIdx(row + 1, col + 1, cols);
                weight = calculateEdgeWeight(image, row, col, row + 1, col + 1);
                edges[num_edges].weight = weight;
                num_edges++;
            }
            // check north-east edges
            if ((row > 0) && (col < cols - 1))
            {
                edges[num_edges].a = p1;
                edges[num_edges].b = flattenedIdx(row - 1, col + 1, cols);
                weight = calculateEdgeWeight(image, row, col, row - 1, col + 1);
                edges[num_edges].weight = weight;
                num_edges++;
            }
        }
    }
}

float calculateTau(float c, int size) { return c / size; }

void GraphSegmentation::segmentGraph(const cv::Mat& image, float c=500)
{
    components.makeset(image);
    calculateEdges(image);
    std::sort(edges, edges + num_edges);

    // calculate thresholds
    float* threshold = new float[rows*cols];
    for (int idx = 0; idx < rows*cols; idx++) { threshold[idx] = calculateTau(c, 1); }

    for (int idx = 0; idx < num_edges; idx++)
    {
        int component1_root = components.findRoot(edges[idx].a);
        int component2_root = components.findRoot(edges[idx].b);

        if (component1_root != component2_root)
        {
            float edge_weight = edges[idx].weight;
            float threshold1 = threshold[component1_root];
            float threshold2 = threshold[component2_root];

            if ((edge_weight < threshold1) && (edge_weight < threshold2))
            {
                components.mergeSets(component1_root, component2_root);
                component1_root = components.findRoot(component1_root);
                threshold[component1_root] = edge_weight + calculateTau(c, components.getSetSize(component1_root));
            }
        }
    }
}
const int GraphSegmentation::getNumEdges()
{
    return num_edges;
}

cv::Mat GraphSegmentation::drawSegments()
{
    cv::Mat canvas = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat color_palette(rows*cols, 1, CV_8UC1);
    cv::randu(color_palette, 0, 255);

    // find number of distinct components
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            int p = flattenedIdx(row, col, cols);
            int component_root = components.findRoot(p);
            uchar color = color_palette.at<uchar>(component_root, 0);
            canvas.at<uchar>(row, col) = color;
        }
    }
    return canvas;
}

