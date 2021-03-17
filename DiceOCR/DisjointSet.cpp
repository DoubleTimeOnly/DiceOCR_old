#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
#include "opencv2/core/core.hpp"
#include "DisjointSet.h"
#include <algorithm>

// how to implement unordered_map with pair as key
// https://www.geeksforgeeks.org/how-to-create-an-unordered_map-of-pairs-in-c/

void DisjointSet::makeset(cv::Mat const& image)
{
    // initialize each pixel in image as its own parent
    rows = image.rows;
    cols = image.cols;

    delete components;
    components = new component[rows*cols];

    int idx;
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            idx = row * cols + col;
            components[idx].parent = idx;
            components[idx].rank = 0;
            components[idx].size = 1;
            components[idx].minX = col;
            components[idx].maxX = col;
            components[idx].minY = row;
            components[idx].maxY = row;
        }
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
    int final_component;
    if (components[p1root].rank < components[p2root].rank)
    {
        components[p1root].parent = p2root;
        components[p2root].size += components[p1root].size;
        final_component = p2root;
    }
    else 
    {
        components[p2root].parent = p1root;
        components[p1root].size += components[p2root].size;
        if (components[p1root].rank == components[p2root].rank)
        {
            components[p1root].rank++;
        }
        final_component = p1root;
    }
    // update bounding box
    components[final_component].minX = std::min(components[p1root].minX, components[p2root].minX);
    components[final_component].minY = std::min(components[p1root].minY, components[p2root].minY);
    components[final_component].maxX = std::max(components[p1root].maxX, components[p2root].maxX);
    components[final_component].maxY = std::max(components[p1root].maxY, components[p2root].maxY);

    num_components--;
}

BoundingBox DisjointSet::getBoundingBoxCoordinates(int root)
{
    BoundingBox boundingbox;
    boundingbox.minX = components[root].minX;
    boundingbox.minY = components[root].minY;
    boundingbox.width = components[root].maxX - components[root].minX;
    boundingbox.height = components[root].maxY - components[root].minY;
    return boundingbox;
}

