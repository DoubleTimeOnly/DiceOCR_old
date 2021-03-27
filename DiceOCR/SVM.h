#pragma once
#include <unordered_map>
#include "opencv2/core/core.hpp"

class SVM
{
private:
    cv::Mat w, dW;
    cv::Mat batchMean, batchStdev;
    cv::Mat affineForward(cv::Mat x);
    cv::Mat forwardPass(cv::Mat x);

public:
    SVM(int inputDims, int numClasses, int rngSeed = 0);
    void train(cv::Mat x, cv::Mat y, int reg);
    int loss(cv::Mat x, cv::Mat y, int reg);
    cv::Mat predict(cv::Mat x);
    void calculateMeanStDev(cv::Mat x);
    void preprocessBatch(cv::Mat x);

    const int getInputDims() { return w.rows; }
    const int getNumClasses() { return w.cols; }
    const cv::Mat getBatchMean() { return batchMean; }
    const cv::Mat getBatchStDev() { return batchStdev; }
};
