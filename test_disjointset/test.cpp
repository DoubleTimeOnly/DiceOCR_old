#include "pch.h"
#include "../DiceOCR/DisjointSet.h"
#include "../DiceOCR/DisjointSet.cpp"
#include "../DiceOCR/GraphSegmentation.h"
#include "../DiceOCR/GraphSegmentation.cpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <filesystem>

//TODO: rename to CreatingDisjointSetFromSingleChannelMatrixShouldHaveAllNodesBeTheirOwnParent
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

// TODO: Rename to CreatingDisjointSetFromLoadedImageShouldHaveExpectedNumberOfPixels
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

//TODO: Rename to CreatingGraphShouldHaveExpectedNumberOfEdges
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


TEST(DisjointSetTests, ComparingEdgesShouldCompareByWeight)
{
    edge edge1, edge2;
    std::pair<int, int> p1(0, 0), p2(1, 1), p3(2, 2);

    edge1.a = p1;
    edge1.b = p2;
    edge1.weight = 2.0;

    edge2.a = p1;
    edge2.b = p3;
    edge2.weight = 1.0;

    EXPECT_FALSE(edge1 < edge2);

    edge1.weight = 1.0;

    EXPECT_FALSE(edge1 < edge2);

    edge1.weight = 0.5;
    EXPECT_TRUE(edge1 < edge2);
}

TEST(DisjointSetTests, SortingEdgeArrayShouldSortArrayByWeight)
{
    edge edge1, edge2, edge3, edge4;
    std::pair<int, int> p1(0, 0), p2(1, 1); 

    edge1.a = p1; edge1.b = p2; edge1.weight = 4.0; 
    edge2.a = p1; edge2.b = p2; edge2.weight = 3.0; 
    edge3.a = p1; edge3.b = p2; edge3.weight = 2.0; 
    edge4.a = p1; edge4.b = p2; edge4.weight = 1.0; 

    edge edges[4] = { edge1, edge2, edge3, edge4 };
    std::sort(edges, edges + 4);

    int previous_weight = 0;
    for (edge e : edges)
    {
        EXPECT_EQ(e.weight - previous_weight, 1);
        previous_weight = e.weight;
    }
}

TEST(DisjointSetTests, SegmentingGraphShouldSegmentGraph)
{
    cv::Mat grayscale_image = cv::imread("./test_disjointset/test_images/d20_grayscale.PNG", cv::IMREAD_GRAYSCALE);
    EXPECT_FALSE(grayscale_image.empty());
    GraphSegmentation segmentationgraph;
    segmentationgraph.segmentGraph(grayscale_image);
    cv::Mat canvas = segmentationgraph.drawSegments();
    cv::imshow("Components", canvas);
    cv::waitKey();
}
