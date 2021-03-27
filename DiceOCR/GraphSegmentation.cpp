#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "GraphSegmentation.h"
#include "DisjointSet.h"
#include <unordered_set>

/*
 Determine the weight between two pixels in an image

 @param image
 @param row, col: coordinates of the pixels to calculate weight between
 @return the weight between two pixels
*/
uchar calculateEdgeWeight(const cv::Mat& image, int row1, int col1, int row2, int col2)
{
    // Mat.at is (y,x)
    int intensity1 = (int)image.at<uchar>(row1, col1);
    int intensity2 = (int)image.at<uchar>(row2, col2);
    uchar weight = std::abs(intensity1 - intensity2);
    return weight;
}

/*
 Return the flattened index of a 2D position

 @param row, col: 2D coordinate in a matrix
 @param max_cols: the number of columns (width) of the matrix
 @return the index of a 2D coordinate if the matrix was flattened
*/
int flattenedIdx(int row, int col, int max_cols)
{
    return row * max_cols + col;
}

/*
 Calculate the 8-connected nearest neighbor edges for each pixel in an image

 @param image
*/
void GraphSegmentation::calculateEdges(const cv::Mat& image)
{
    rows = image.rows;
    cols = image.cols;
    delete edges;
    edges = new edge[rows*cols*4];    // create array of edges

    num_edges = 0;
    int p1;
    uchar weight;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
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

/*
 Calculate the threshold Tau used to help determine if two disjoint sets should be merged

 @param c: a hyperparameter. Higher values mean components can have larger weights between them and still be merged, which leads to larger final segmentations
 @param size: the size of a disjoint set
*/
float calculateTau(float c, int size) { return c / size; }

bool operator<(const edge& a, const edge& b) { return a.weight < b.weight; }

/*
 Segment an image using graph-based segmentation

 @param image: image to segment
 @param c: hyperparameter. Larger values mean larger final segmentations.
 @param minsize: the smallest area in pixels a segmentation can be
*/
void GraphSegmentation::segmentGraph(const cv::Mat& image, float c, int minsize)
{
    components.makeset(image);
    calculateEdges(image);
    std::sort(edges, edges + num_edges);

    // thresholds are used to determine if two components should be merged
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
    delete threshold;

    // Post processing removes very small segmentations
    for (int i = 0; i < num_edges; i++)
    {
        int component1_root = components.findRoot(edges[i].a);
        int component2_root = components.findRoot(edges[i].b);

        if (component1_root != component2_root)
        {
            int component1_size = components.getSetSize(component1_root);
            int component2_size = components.getSetSize(component2_root);
            if ((component1_size < minsize) || (component2_size < minsize)) 
            {
                components.mergeSets(component1_root, component2_root);
            }
        }
    }
}

const int GraphSegmentation::getNumEdges()
{
    return num_edges;
}

/*
 Visualize the final segmentations

 @param drawboxes: if true, draw the bounding boxes around each segmentation
 @return an image that shows each segmentation and their bounding boxes
*/
cv::Mat GraphSegmentation::drawSegments(bool drawboxes)
{
    cv::Mat canvas = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat color_palette(rows*cols, 1, CV_8UC1);
    cv::randu(color_palette, 0, 255);

    int p, component_root;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            p = flattenedIdx(row, col, cols);
            component_root = components.findRoot(p);
            uchar color = color_palette.at<uchar>(component_root, 0);
            canvas.at<uchar>(row, col) = color;
        }
    }

    if (drawboxes)
    {
        std::vector<cv::Rect> regions;
        getROIs(regions, cols, rows);
        for (const cv::Rect& region : regions)
        {
            cv::rectangle(canvas, region, cv::Scalar(255));
        }
    }
    return canvas;
}

/*
 Get the axis-aligned bounding boxes of each segmented component

 @param regions: vector to pass the found regions to
 @param maxWidth, maxHeight: max width and height of component bounding boxes
*/
void GraphSegmentation::getROIs(std::vector<cv::Rect>& regions, int maxWidth, int maxHeight) 
{
    std::unordered_set<int> visited_components = {};
    int component_root;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            component_root = components.findRoot(flattenedIdx(row, col, cols));
            if (visited_components.find(component_root) == visited_components.end()) 
            { 
                cv::Rect region = components.getBoundingBoxCoordinates(component_root);
                if ((region.width < maxWidth) && (region.height < maxHeight))
                {
                    regions.push_back(region);
                }
                visited_components.insert(component_root); 
            }
        }
    }
}
