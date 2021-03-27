#include "opencv2/core/core.hpp"
#include "SVM.h"

SVM::SVM(int inputDims, int numClasses, int rngSeed)
{
    if (rngSeed != 0) 
    {
        cv::theRNG().state = rngSeed;
    }
    w = cv::Mat(inputDims + 1, numClasses, CV_32F) * 0.0001;
    dW = cv::Mat(inputDims, numClasses, CV_32F);
    cv::randn(w, 0, 0.1);
}

/*
 Mean center and normalization

 @param x: mini-batch of shape (N, D)
*/
void SVM::calculateMeanStDev(cv::Mat x)
{
    cv::Mat x_processed;
    cv::reduce(x, batchMean, 0, CV_REDUCE_AVG, CV_32F);
    for (int example = 0; example < x.rows; example++)
    {
        x.row(example) -= batchMean;
    }
    // x_processed = x - batchMean;

    batchStdev = x.mul(x);
    cv::reduce(batchStdev, batchStdev, 0, CV_REDUCE_SUM, CV_32F);
    batchStdev /= x.rows;
    cv::sqrt(batchStdev, batchStdev);
}

/* 
 subtract training batch mean and divide by stdev

 @param x: mini-batch of shape (N, D)
*/
void SVM::preprocessBatch(cv::Mat x)
{
    for (int example = 0; example < x.rows; example++)
    {
        x.row(example) -= batchMean;
        x.row(example) /= batchStdev;
    }
    cv::Mat biasTerm(x.rows, 1, x.type());
    biasTerm = cv::Scalar(1);
    cv::hconcat(x, biasTerm, x);
}

cv::Mat SVM::affineForward(cv::Mat x)
{
    x.convertTo(x, CV_32F);
    return cv::Mat(x*w);
}

cv::Mat SVM::forwardPass(cv::Mat x )
{
    cv::Mat scores = affineForward(x);
    return scores;
}

/*
 calculate the multi-class hinge loss of an SVM

 @param x: matrix of shape (N, D)
 @param w: matrix of shape (D, C)
 @param y: correct class label of input x
 @param reg: regularization strength
*/
int SVM::loss(cv::Mat x, cv::Mat y, int reg)
{
    cv::Mat scores = forwardPass(x);
    return 0;
}
