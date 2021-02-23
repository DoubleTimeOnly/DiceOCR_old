#include <iostream>
#include <vector>
#include <unordered_map>

class DisjointSet
{
    std::unordered_map<int, int> parent;    // stores each node in set and their parent
    std::unordered_map<int, int> rank;    // store the rank of each node

public:
    
    void makeset(std::vector<int> const& universe)
    {
        // initialize each value in vector as its own parent
        for (int i : universe) {
            parent[i] = i;
            rank[i] = 0;
        }
    }

    int findRoot(int k)
    {
        if (parent[k] != k)
        {
            parent[k] = findRoot(parent[k]);
        }
        return parent[k];
    }

    void mergeSets(int x, int y)
    {
        int xRoot = findRoot(x);
        int yRoot = findRoot(y);

        // merge the two sets based on their rank / depth
        // larger ranked set consunmes the lower ranked one
        // if they are the same, the depth has effectively increased
        if (rank[xRoot] > rank[yRoot])
        {
            parent[yRoot] = xRoot;
        }
        else if (rank[xRoot] < rank[yRoot])
        {
            parent[xRoot] = yRoot;
        }
        else
        {
            parent[xRoot] = yRoot;
            rank[xRoot] += 1;
        }
    }
};