#include "opencv2/core/core.hpp"
#include <iostream>
#include "ImageProcessing.h"

cv::Mat toGrayscale(cv::Mat& src) {
    int src_channels = src.channels();

    cv::Mat grayscale_img;
    if (src_channels == 1) {
        grayscale_img = src.clone();
    }
    else if (src_channels == 3) {
        int src_rows = src.rows;
        int src_cols = src.cols;
        grayscale_img.create(src_rows, src_cols, CV_8U);
        uchar r, g, b;
        for (int row = 0; row < src_rows; row++) {
            // get pointers to start of image rows
            cv::Vec3b* src_pixel = src.ptr<cv::Vec3b>(row);
            uchar* grayscale_pixel = grayscale_img.ptr<uchar>(row);
            for (int col = 0; col < src_cols; col++) {
                r = src_pixel[col][2];
                g = src_pixel[col][1];
                b = src_pixel[col][0];
                grayscale_pixel[col] = (uchar)(0.299 * r + 0.587 * g + 0.114 * b);
            }
        }
    }
    else {
        throw "Invalid number of channels in src image in function toGrayscale";
    }
    return grayscale_img;
}

cv::Mat threshold(cv::Mat& src, int threshold){
    int src_channels = src.channels();

    if (src_channels == 1) {
        int src_rows = src.rows;
        int src_cols = src.cols;
        for (int row = 0; row < src_rows; row++) {
            uchar* src_pixel = src.ptr<uchar>(row);
            for (int col = 0; col < src_cols; col++) {
                if (src_pixel[col] < threshold) {
                    src_pixel[col] = 0;
                }
                else if (src_pixel[col] >= threshold) {
                    src_pixel[col] = 255;
                }
            }
        }
    }
    else {
        throw "Invalid number of channels in src image in function threshold";
    }
    return src;
}