#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "CaptureImage.h"
#include "ImageProcessing.h"

cv::VideoCapture g_cap;

#include <opencv2/core/core.hpp>
int main() {
    std::cout << "Hello World" << std::endl;
    cv::Mat frame;
    cv::Mat grayscale;
    cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);
    g_cap.open(0);

    for (;;) {
        g_cap >> frame;
        //threshold(frame, 125);
        grayscale = toGrayscale(frame);
        cv::Mat thresholded_image = threshold(grayscale, 128);
        cv::imshow("Webcam", thresholded_image);
        cv::waitKey(1);
    }
    return 1;
}