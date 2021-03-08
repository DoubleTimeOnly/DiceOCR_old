#pragma once
#include <unordered_map>
#include "opencv2/core/core.hpp"

struct hash_pair {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ hash2;
    }
};

struct component
{
    int parent;
    int rank;
    int size;
};

class DisjointSet
{
private:
    int rows;
    int cols;
    int num_components;
    component* components = NULL;

public:
    std::unordered_map<cv::Point2i, cv::Point2i,  hash_pair> parent;    // stores each node in set and their parent
    std::unordered_map<std::pair<int, int>, int, hash_pair> rank;    // store the rank of each node
    std::unordered_map<std::pair<int, int>, int, hash_pair> setsize;    // store the size of each node
    void makeset(cv::Mat const& image);
    int findRoot(int p);
    void mergeSets(int p1, int p2);

    const size_t getSetSize();
    const int findRank(int p) { return this->components[findRoot(p)].rank; }
    cv::Mat drawSegments();
    const int getNumComponents() { return this->num_components; }
    const int getSetSize(int p) { return this->components[p].size; }
    
};



