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

class DisjointSet
{
public:
    std::unordered_map<std::pair<int, int>, std::pair<int, int>,  hash_pair> parent;    // stores each node in set and their parent
    std::unordered_map<std::pair<int, int>, int, hash_pair> rank;    // store the rank of each node
    void makeset(cv::Mat const& image);
    std::pair<int, int> findRoot(std::pair<int, int> p);
    void mergeSets(std::pair<int, int> p1, std::pair<int, int> p2);

    const size_t getSetSize();
    const int findRank(const std::pair<int, int>& p);
};



