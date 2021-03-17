#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
#include "opencv2/core/core.hpp"
#include "DisjointSet.h"
#include <algorithm>

/*
 Initialize each pixel in an image as its own disjoint set
 
 @param image: image to create disjoint sets from
*/
void DisjointSet::makeset(cv::Mat const& image)
{
    rows = image.rows;
    cols = image.cols;

    delete components;
    // The disjoint sets are stored in a flattened array of the same size as the image
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

/*
 Given the element of a disjoint set, find the representative (root) element of said set

 @param p: an element of a disjoint set
 @return the representative (root) element of the disjoint set p is a part of
*/
int DisjointSet::findRoot(int p)
{
    // compress sets to make finding root faster in the future
    if (components[p].parent != p)
    {
        components[p].parent = findRoot(components[p].parent);
    }
    return components[p].parent;
}

/*
 Merge two disjoint sets

 Merging a set means pointing the parent of the smaller set to the parent of the larger set
 and updating the set's properties. Elements of both sets will now point to the same root.

 @params p1, p2: elements of disjoint sets
*/
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

/*
 Get the bounding box of a disjoint set given its root

 @param root: the representative (root) element of a disjoint set
 @return a cv::Rect that represents the bounding box that encompasses the disjoint set
*/
const cv::Rect DisjointSet::getBoundingBoxCoordinates(int root)
{
    return cv::Rect(
        components[root].minX,
        components[root].minY,
        components[root].maxX - components[root].minX,
        components[root].maxY - components[root].minY
    );
}

