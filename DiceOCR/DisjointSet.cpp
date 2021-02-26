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
    int rows = image.rows;
    int cols = image.cols;

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++) 
        {
            std::pair<int, int> p(col, row);
            parent[p] = p;
            rank[p] = 0;
        }
    }
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
        parent[p1root] = p2root;
        rank[p1root] += 1;
    }
}

const size_t DisjointSet::getSetSize()
{
    return parent.size();
}
