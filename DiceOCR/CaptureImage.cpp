#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "CaptureImage.h"
#include "ImageProcessing.h"
#include "GraphSegmentation.h"

cv::VideoCapture g_cap;

#include <opencv2/core/core.hpp>
int main() {
    std::cout << "Hello World" << std::endl;
    cv::Mat frame;
    cv::Mat grayscale;
    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);
    g_cap.open(0);
    GraphSegmentation segmenter;

    for (;;) {
        g_cap >> frame;
        //threshold(frame, 125);
        grayscale = toGrayscale(frame);
        segmenter.segmentGraph(grayscale, 2000, 100);
        cv::Mat segmented_image = segmenter.drawSegments();
        cv::imshow("Segmented Image", segmented_image);
        cv::imshow("Webcam", grayscale);
        cv::waitKey(1);
    }
    return 1;
}