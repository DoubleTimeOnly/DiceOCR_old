#include "pch.h"
#include "../DiceOCR/DisjointSet.h"
#include "../DiceOCR/DisjointSet.cpp"
#include "../DiceOCR/GraphSegmentation.h"
#include "../DiceOCR/GraphSegmentation.cpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <filesystem>


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

TEST(DisjointSetTests, CreatingDisjointSetFromLoadedImageShouldRaiseNoErrors)
{
    cv::Mat grayscale_image = cv::imread("./test_disjointset/test_images/d20_grayscale.PNG", cv::IMREAD_GRAYSCALE);
    EXPECT_FALSE(grayscale_image.empty());
    EXPECT_EQ(grayscale_image.rows, 479);
    EXPECT_EQ(grayscale_image.cols, 639);
    DisjointSet disjointset;
    disjointset.makeset(grayscale_image);
    int num_pixels = grayscale_image.rows * grayscale_image.cols;
    EXPECT_EQ(disjointset.getSetSize(), num_pixels);
}

TEST(DisjointSetTests, CreatingGraphShouldRaiseNoErrors)
{
    cv::Mat grayscale_image = cv::imread("./test_disjointset/test_images/d20_grayscale.PNG", cv::IMREAD_GRAYSCALE);
    EXPECT_FALSE(grayscale_image.empty());
    GraphSegmentation segmentationgraph;
    segmentationgraph.calculateEdges(grayscale_image);

    int rows = grayscale_image.rows;
    int cols = grayscale_image.cols;
    int expected_num_edges = 4 * cols*rows - 3 * (rows + cols) + 2;
    EXPECT_EQ(expected_num_edges, segmentationgraph.getNumEdges());
}
