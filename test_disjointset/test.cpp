#include "pch.h"
#include "../DiceOCR/DisjointSet.h"
#include "../DiceOCR/DisjointSet.cpp"
#include "opencv2/core/core.hpp"

TEST(DisjointSetTests, CreatingDisjointSetFromGrayscaleImageShouldHaveNoErrors) {
    DisjointSet disjointset;
    cv::Mat grayscale_image = cv::Mat::ones(2, 2, CV_8U);
    disjointset.makeset(grayscale_image);

    EXPECT_EQ(disjointset.getSetSize(), 4);
    EXPECT_TRUE(true);
}