#include "pch.h"
#include "../DiceOCR/DisjointSet.h"
#include "../DiceOCR/DisjointSet.cpp"
#include "opencv2/core/core.hpp"

TEST(DisjointSetTests, CreatingDisjointSetFromGrayscaleImageShouldHaveNoErrors) {
    DisjointSet disjointset;
    cv::Mat grayscale_image = cv::Mat::ones(2, 2, CV_8U);
    disjointset.makeset(grayscale_image);

    EXPECT_EQ(disjointset.getSetSize(), 4);

    // check parents set properly
    for (int row = 0; row < grayscale_image.rows; row++)
    {
        for (int col = 0; col < grayscale_image.cols; col++)
        {
            if (row == 0 && col == 0)
            {
                continue;
            }
            std::pair<int, int> p(row, col);
            std::pair<int, int> parent = disjointset.findRoot(p);
            EXPECT_EQ(parent, p);
        }
    }
}

TEST(DisjointSetTests, MergingTwoDisjointSetsShouldHaveNoErrors)
{
    DisjointSet disjointset;
    cv::Mat grayscale_image = cv::Mat::ones(2, 2, CV_8U);
    disjointset.makeset(grayscale_image);
    
    std::pair<int, int> origin(0, 0);
    // check set merging and path compression
    for (int row = 0; row < grayscale_image.rows; row++)
    {
        for (int col = 0; col < grayscale_image.cols; col++)
        {
            if (row == 0 && col == 0)
            {
                continue;
            }
            std::pair<int, int> p(row, col);
            disjointset.mergeSets(origin, p);
            std::pair<int, int> parent = disjointset.findRoot(p);
            EXPECT_EQ(origin, parent);
        }
    }
}

TEST(DisjointSetTests, MergingTwoEqualRankSetsShouldIncreaseTheRankOfBothSets)
{
    DisjointSet disjointset;
    cv::Mat grayscale_image = cv::Mat::ones(2, 1, CV_8U);
    disjointset.makeset(grayscale_image);

    std::pair<int, int> p0(0, 0);
    std::pair<int, int> p1(1, 0);
    EXPECT_EQ(disjointset.findRank(p0), 0);
    EXPECT_EQ(disjointset.findRank(p1), 0);
    disjointset.mergeSets(p0, p1);
    EXPECT_EQ(disjointset.findRank(p0), 1);
    EXPECT_EQ(disjointset.findRank(p1), 1); // root of p1 is p0, so its rank should be the same
}
