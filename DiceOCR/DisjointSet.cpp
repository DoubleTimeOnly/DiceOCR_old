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

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++) 
        {
            std::pair<int, int> p(col, row);
            parent[p] = p;
            rank[p] = 0;
        }
    }
    num_components = rows * cols;
}

std::pair<int, int> DisjointSet::findRoot(std::pair<int, int> p)
{
    // path compression
    // set parent of each node to the root of its respective set
    if (parent[p] != p)
    {
        parent[p] = findRoot(parent[p]);
    }
    return parent[p];
}

void DisjointSet::mergeSets(std::pair<int, int> p1, std::pair<int, int> p2)
{
    std::pair<int, int> p1root = findRoot(p1);
    std::pair<int, int> p2root = findRoot(p2);

    // merge the two sets based on their rank / depth
    // larger ranked set consunmes the lower ranked one
    // if they are the same, the depth has effectively increased
    if (rank[p1root] > rank[p2root])
    {
        parent[p2root] = p1root;
    }
    else if (rank[p1root] < rank[p2root])
    {
        parent[p1root] = p2root;
    }
    else
    {
        parent[p2root] = p1root;
        rank[p1root]++;
    }
    num_components--;
}

const size_t DisjointSet::getSetSize()
{
    return parent.size();
}

const int DisjointSet::findRank(const std::pair<int, int>& p)
{
    return rank[findRoot(p)];
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
            std::pair<int, int> p(col, row);
            std::pair<int, int> component_root = findRoot(p);
            //std::unordered_map<std::pair<int, int>, int>::const_iterator color_location = color_palette.find(component_root);

            //int component_color;
            //if (color_location == color_palette.end())
            //{
            //    color_palette[component_root] = colors.back();
            //    component_color = colors.back();
            //    colors.pop_back();
            //}
            //else
            //{
            //    component_color = color_palette[component_root];
            //}
            uchar color = color_palette.at<uchar>(component_root.second, component_root.first);
            canvas.at<uchar>(row, col) = color;
        }
    }
    return canvas;
}
