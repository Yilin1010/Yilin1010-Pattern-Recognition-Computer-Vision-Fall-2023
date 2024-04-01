/*
  Author: Yilin Tang
  Date: 2024-02-24
  CS 5330 Computer Vision
  Description: 

*/


#ifndef ORalgorithms_H
#define ORalgorithms_H
#include "distance.h"


static const std::string unknownClass("unknown");
// declarations here
enum class FeatureSize
{
  fixedSize,
  variableSize
};

enum class ClassificationMethod {
    NN, 
    KNN
};

uchar meanDominantcolor(cv::Mat &src, int k = 2);
int dynamicThreshold(cv::Mat &src, cv::Mat &dst);
int cleanupBinary(cv::Mat &src, cv::Mat &dst);
int segmentIntoRegion(cv::Mat &src, cv::Mat &dst, std::vector<std::vector<cv::Point>> &rectBoxs,int maxComponets = 4);
int leastInertiaInvariance(cv::Mat &src, cv::Mat &dst, std::vector<std::vector<cv::Point>> &Pointsets, std::vector<std::vector<float>> &featureMap);
int extractFeatureVector(cv::Mat &src,std::vector<float> &featureVec,FeatureSize);
std::string classify(cv::Mat &target, const char *objectDBfilename, DistanceMetric metric = DistanceMetric::EuclideanDistance, ClassificationMethod method = ClassificationMethod::KNN );


#endif // ORalgorithms_H