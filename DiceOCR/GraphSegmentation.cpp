#include "opencv2/core/core.hpp"
#include "GraphSegmentation.h"
#include "DisjointSet.h"

float calculateEdgeWeight(const cv::Mat& image, const std::pair<int, int>& p1, const std::pair<int, int>& p2)
{
    // Mat.at is (y,x)
    int intensity1 = image.at<uchar>(p1.second, p1.first);
    int intensity2 = image.at<uchar>(p2.second, p2.first);
    float weight = std::abs(intensity1 - intensity2);
    return weight;
}

edge* GraphSegmentation::calculateEdges(const cv::Mat& image)
{
    const int rows = image.rows;
    const int cols = image.cols;
    edges = new edge[rows*cols*4];    // create array of edges

    num_edges = 0;
    component_weights = cv::Mat::zeros(rows, cols, CV_32FC1);
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            float max_weight = 0.0;
            std::pair<int, int> p1(col, row);
            // check vertical edges
            if (row < rows - 1)
            {
                edges[num_edges].a = p1;
                std::pair<int, int> p2(col, row + 1);
                edges[num_edges].b = p2;
                float weight = calculateEdgeWeight(image, p1, p2);
                edges[num_edges].weight = weight;

                if (weight > max_weight) { max_weight = weight; }
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

                if (weight > max_weight) { max_weight = weight; }
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

                if (weight > max_weight) { max_weight = weight; }
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

                if (weight > max_weight) { max_weight = weight; }
                num_edges++;
            }
            component_weights.at<float>(row, col) = max_weight;
        }
    }
    return edges;

}

void GraphSegmentation::segmentGraph(const cv::Mat& image)
{
    components.makeset(image);
    calculateEdges(image);
    std::sort(edges, edges + num_edges);

    //int previous_num_components = 0;
    //while (components.getNumComponents() != previous_num_components)
    //{
        //previous_num_components = components.getNumComponents();
        for (int idx = 0; idx < num_edges; idx++)
        {
            std::pair<int, int> component1_root = components.findRoot(edges[idx].a);
            std::pair<int, int> component2_root = components.findRoot(edges[idx].b);

            if (component1_root != component2_root)
            {
                float edge_weight = edges[idx].weight;
                int type = component_weights.type();
                float component1_weight = component_weights.at<float>(component1_root.second, component1_root.first);
                float component2_weight = component_weights.at<float>(component2_root.second, component2_root.first);

                if ((edge_weight < component1_weight) && (edge_weight < component2_weight))
                {
                    components.mergeSets(component1_root, component2_root);
                    float max_weight = std::max(component1_weight, component2_weight);
                    component_weights.at<float>(component1_root.second, component1_root.first) = max_weight;
                    component_weights.at<float>(component2_root.second, component2_root.first) = max_weight;
                }
            }
        }
    //}

}
const int GraphSegmentation::getNumEdges()
{
    return num_edges;
}

cv::Mat GraphSegmentation::drawSegments()
{
    return components.drawSegments();
}

