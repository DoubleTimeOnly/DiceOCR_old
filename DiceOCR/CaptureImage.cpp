#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "CaptureImage.h"
#include "ImageProcessing.h"
#include "GraphSegmentation.h"

cv::VideoCapture g_cap;

#include <opencv2/core/core.hpp>
int main() {
    bool useWebcam = false;

    if (useWebcam)
    {
        cv::Mat frame;
        cv::Mat grayscale;
        cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);
        g_cap.open(0);
        GraphSegmentation segmenter;

        for (;;) {
            g_cap >> frame;
            grayscale = toGrayscale(frame);
            segmenter.segmentGraph(grayscale, 20000, 100);
            cv::Mat segmented_image = segmenter.drawSegments();
            cv::imshow("Segmented Image", segmented_image);
            cv::imshow("Webcam", grayscale);
            cv::waitKey(1);
        }
    }
    else
    {
         cv::Mat image = cv::imread("../test_disjointset/test_images/catandice2.jpg", cv::IMREAD_GRAYSCALE);
        //cv::Mat image = cv::imread("../test_disjointset/test_images/d20_grayscale.PNG", cv::IMREAD_GRAYSCALE);
        if (image.empty())
        {
            std::cout << "Could not read image" << std::endl;
            return 0;
        }
        cv::GaussianBlur(image, image, cv::Size(5, 5), 1);
        GraphSegmentation segmenter;
        segmenter.segmentGraph(image, 3000, 100);
        cv::Mat segmented_image = segmenter.drawSegments(false);
        cv::hconcat(image, segmented_image, image);
        cv::imshow("image", image);
        cv::waitKey();
    }
    return 1;
}