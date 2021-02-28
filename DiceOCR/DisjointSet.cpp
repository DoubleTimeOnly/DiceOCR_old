#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
#include "opencv2/core/core.hpp"
#include "DisjointSet.h"

// how to implement unordered_map with pair as key
// https://www.geeksforgeeks.org/how-to-create-an-unordered_map-of-pairs-in-c/

void DisjointSet::makeset(cv::Mat const& image)
{
    // initialize each pixel in image as its own parent
    rows = image.rows;
    cols = image.cols;

    components = new component[rows*cols];

    for (int idx = 0; idx < rows*cols; idx++)
    {
        components[idx].parent = idx;
        components[idx].rank = 0;
        components[idx].size = 1;
    }
    num_components = rows * cols;
}

int DisjointSet::findRoot(int p)
{
    // path compression
    // set parent of each node to the root of its respective set
    if (components[p].parent != p)
    {
        components[p].parent = findRoot(components[p].parent);
    }
    return components[p].parent;
}

void DisjointSet::mergeSets(int p1, int p2)
{
    int p1root = findRoot(p1);
    int p2root = findRoot(p2);

    // merge the two sets based on their rank / depth
    // larger ranked set consunmes the lower ranked one
    // if they are the same, the depth has effectively increased
    if (components[p1root].rank < components[p2root].rank)
    {
        components[p1root].parent = p2root;
        components[p2root].size += components[p1root].size;
    }
    else 
    {
        components[p2root].parent = p1root;
        components[p1root].size += components[p2root].size;
        if (components[p1root].rank == components[p2root].rank)
        {
            components[p1root].rank++;
        }
    }
    num_components--;
}

cv::Mat DisjointSet::drawSegments()
{
    cv::Mat canvas = cv::Mat::zeros(rows, cols, CV_8UC1);
    cv::Mat color_palette(rows, cols, CV_8UC1);
    cv::randu(color_palette, 0, 255);

    // find number of distinct components
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            int p = row * rows + col;
            int component_root = findRoot(p);
            //uchar color = color_palette.at<uchar>(component_root.second, component_root.first);
            //canvas.at<uchar>(row, col) = color;
        }
    }
    return canvas;
}
