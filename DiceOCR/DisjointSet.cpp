#include <iostream>
#include <vector>
#include <unordered_map>
#include "opencv2/core/core.hpp"

class DisjointSet
{
    std::unordered_map<cv::Point2i, cv::Point2i> parent;    // stores each node in set and their parent
    std::unordered_map<cv::Point2i, int> rank;    // store the rank of each node

public:
    
    void makeset(cv::Mat const& image)
    {
        // initialize each pixel in image as its own parent
        int rows = image.rows;
        int cols = image.cols;

        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++) 
            {
                cv::Point2i p(col, row);
                parent[p] = p;
                rank[p] = 0;
            }
        }
    }

    cv::Point2i findRoot(cv::Point2i p)
    {
        // path compression
        // set parent of each node to the root of its respective set
        if (parent[p] != p)
        {
            parent[p] = findRoot(parent[p]);
        }
        return parent[p];
    }

    void mergeSets(cv::Point2i p1, cv::Point2i p2)
    {
        cv::Point2i p1root = findRoot(p1);
        cv::Point2i p2root = findRoot(p2);

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
};