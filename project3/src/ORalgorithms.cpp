/*
  Author: Yilin Tang
  Date: 2024-02-24
  CS 5330 Computer Vision
  Description:

  provide preprocess methods for object recogniton,
  Thresholding: separates objects from the background.
  Cleanup: use morphological operations to refine object shapes.
  Region Segmentation: indentify object regions.
  Feature Extraction: analyze region properties.

  provide Nearest Neighbor (NN) and K-Nearest Neighbors (KNN),
  scaled Euclidean distance and cosine distance.
  sum of distance, mean of distance, weighted voting.

*/

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <iterator> // For std::ostream_iterator
#include <fstream>
#include <filesystem>
#include <stdio.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "csv_util.h"
#include "distance.h"
#include "ORalgorithms.h"

using namespace cv;
using namespace std;

RNG rng(12345);

std::vector<cv::Vec3b> colorList = {
    cv::Vec3b(214, 112, 218),
    cv::Vec3b(87, 139, 46),
    cv::Vec3b(144, 128, 112),
    cv::Vec3b(235, 206, 135),
};
/**
 * @brief kmeans dominiate color analysis
 *
 */
uchar meanDominantcolor(cv::Mat &src, int k)
{
  Mat data;
  src.convertTo(data, CV_32F);
  data = data.reshape(1, data.total());

  // k-means clustering
  Mat labels, centers;
  kmeans(data, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

  centers = centers.reshape(3, centers.rows);
  centers.convertTo(centers, CV_8U);

  vector<Vec3b> dominantColors;
  for (int i = 0; i < centers.rows; i++)
  {
    dominantColors.push_back(centers.at<Vec3b>(i, 0));
  }

  Vec3i sum(0, 0, 0);
  for (const Vec3b &color : dominantColors)
  {
    sum[0] += color[0];
    sum[1] += color[1];
    sum[2] += color[2];
  }

  sum[0] /= dominantColors.size();
  sum[1] /= dominantColors.size();
  sum[2] /= dominantColors.size();

  // get grayvalue of mean of 2 dominant color
  Vec3b meancolor = Vec3b(sum[0], sum[1], sum[2]);
  Mat meanColorMat(1, 1, CV_8UC3, meancolor);
  Mat meanGray;
  cvtColor(meanColorMat, meanGray, COLOR_BGR2GRAY);
  uchar meanGrayVal = meanGray.at<uchar>(0, 0);

  return meanGrayVal;
}

/**
 * @brief
 * calculate mean of grayscale K-means dominate color
 * @param src
 * @param dominantgrayColors
 * @param k
 * @return uchar
 */
uchar meangrayDominantcolor(cv::Mat &src, vector<uchar> &dominantgrayColors, int k)
{
  Mat data;
  src.convertTo(data, CV_32F);
  data = data.reshape(1, data.total());

  // k-means clustering
  Mat labels, centers;
  kmeans(data, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

  centers = centers.reshape(3, centers.rows);
  centers.convertTo(centers, CV_8U);

  int GrayValsum = 0;
  for (int i = 0; i < centers.rows; i++)
  {
    // get grayvalue of mean of dominant color
    Mat colorMat(1, 1, CV_8UC3, centers.at<Vec3b>(i, 0));
    Mat graydominant;
    cvtColor(colorMat, graydominant, COLOR_BGR2GRAY);
    uchar grayVal = graydominant.at<uchar>(0, 0);
    dominantgrayColors.push_back(grayVal);
    GrayValsum += grayVal;
  }
  GrayValsum /= dominantgrayColors.size();
  return static_cast<uchar>(GrayValsum);
}

int customthreshold(cv::Mat &src, cv::Mat &dst, float threshold, float maxValue, bool aboveToMax)
{
  if (src.channels() != 1)
  {
    printf("image must be sinble channel");
    return 1;
  }

  dst = cv::Mat::zeros(src.size(), src.type());

  for (int row = 0; row < src.rows; ++row)
  {
    for (int col = 0; col < src.cols; ++col)
    {
      // Get the current pixel value
      uchar pixelValue = src.at<uchar>(row, col);
      if (aboveToMax)
      {
        // above set to maxVal, otherwise, 0
        dst.at<uchar>(row, col) = pixelValue > threshold ? static_cast<uchar>(maxValue) : 0;
      }
      else
      {
        // below or equal set to maxVal, otherwise, 0
        dst.at<uchar>(row, col) = pixelValue > threshold ? 0 : static_cast<uchar>(maxValue);
      }
    }
  }
  return 0;
}

/**
 * @brief
 * dynamic threshold to indentity objects and backgroud
 * @param src
 * @param dst
 * @return int
 */
int dynamicThreshold(cv::Mat &src, cv::Mat &dst)
{
  // pre-processing image
  cv::Mat gray;
  cv::cvtColor(src, gray, COLOR_RGB2GRAY);

  // uchar meangrayDominant = meanDominantcolor(src);
  vector<uchar> dominantgrayColors;
  uchar meangrayDominant = meangrayDominantcolor(src, dominantgrayColors, 2);
  uchar moreWhitedominant = std::max(dominantgrayColors[0], dominantgrayColors[1]);
  uchar darkerdominant = std::min(dominantgrayColors[0], dominantgrayColors[1]);
  //  meanColor aproximately show backgroud color
  uchar approxBackgroudColor = cv::mean(gray)[0];

  //  we need to set backgroud to black
  //  if backgroud is more darker, thresholds, below set to 0(black)
  if (approxBackgroudColor - darkerdominant < moreWhitedominant - approxBackgroudColor)
  {
    // lower threshold from mean to eliminate large white area
    // cv::threshold(gray, dst, meangrayDominant, 255, cv::THRESH_BINARY);
    customthreshold(gray, dst, meangrayDominant, 255, true);
  }
  else
  {
    // if backgroud is more white, thresholds, above set to 0(black)
    // cv::threshold(gray, dst, meangrayDominant, 255, cv::THRESH_BINARY_INV);
    customthreshold(gray, dst, meangrayDominant, 255, false);
  }
  return (0);
}

/**
 * @brief
 * this operation can take in-place
 * @param src
 * @param dst
 * @return int
 */
int cleanupBinary(cv::Mat &src, cv::Mat &dst)
{
  // morphology filters can be used in place
  src.copyTo(dst);

  // larger morph kernel will connect parts of an object that are far apart
  // smaller will remove small noise without affecting structure
  int NoiseSize = 3;
  int connectSize = 2;
  cv::Mat Noisekernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * NoiseSize + 1, 2 * NoiseSize + 1),
                                                  cv::Point(NoiseSize, NoiseSize));

  cv::Mat connectkernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * connectSize + 1, 2 * connectSize + 1),
                                                    cv::Point(connectSize, connectSize));

  // cv::erode(src, dst, kernel);

  // connect unconnected parts
  cv::dilate(dst, dst, connectkernel);

  // removing outside noise
  cv::morphologyEx(dst, dst, cv::MORPH_OPEN, Noisekernel);
  // removing small holes
  // cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, kernel);

  return (0);
}

/**
 * @brief
 * @param src binary image where
 * the foreground is white (255) and the background is black (0).
 * @param dst 3-channel BGR image to
 * display colored regions and bounding boxes.
 * @param Pointsets
 * @param maxComponets default 4
 * @return int
 */
int segmentIntoRegion(cv::Mat &src, cv::Mat &dst, std::vector<std::vector<cv::Point>> &Pointsets, int maxComponets)
{

  // clear previous points to avoid duplicates
  // relocate for at most N components
  Pointsets.clear();
  std::vector<std::pair<int, std::vector<cv::Point>>> areaPointsets;

  // draw display grayimage as 3-channel image
  cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);

  int minSize = src.rows * src.cols / 300;
  // get a region map
  cv::Mat labels, stats, centroids;
  int nlabels = connectedComponentsWithStats(src, labels, stats, centroids, 4, CV_16U);

  // compute region size, region centroid, and region axis-oriented bounding box
  int left, top, width, height, area;

  // std::vector<ushort> selectedLabels;
  // std::vector<Vec3b> regionColors;

  // skip background,  usually label 0
  for (ushort label = 1; label < nlabels; ++label)
  {
    // ignore region touching boundary
    // remove small region
    // maybe pick central region

    left = stats.at<int>(label, cv::CC_STAT_LEFT);
    top = stats.at<int>(label, cv::CC_STAT_TOP);
    width = stats.at<int>(label, cv::CC_STAT_WIDTH);
    height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
    area = stats.at<int>(label, cv::CC_STAT_AREA);

    // Check if the region is large enough
    if (area < minSize)
      continue;

    // Check if the region touches the image boundary
    // this could loss major area
    if (left == 0 || top == 0 || (left + width) == src.cols || (top + height) == src.rows)
      continue;

    // TODO: match last image centroid and color
    cv::Vec3b color = colorList[areaPointsets.size()];

    // draw douding box for region
    cv::Point pt1(left, top);
    cv::Point pt2(left + width, top + height);
    // define color and thickness
    cv::rectangle(dst, pt1, pt2, color, 2);

    // draw centriods
    double x = centroids.at<double>(label, 0);
    double y = centroids.at<double>(label, 1);
    cv::circle(dst, cv::Point(x, y), 4, color, 2);

    std::vector<cv::Point> points;
    // display each selected regin with different colors
    // !!Reminder: do not access ushort(16U) as int(32S), it will cause error
    // !!Reminder: point(col,row)
    for (int row = 0; row < labels.rows; ++row)
    {
      for (int col = 0; col < labels.cols; ++col)
      {
        ushort cur_label = labels.at<ushort>(row, col);
        if (cur_label == label)
        {
          // save componets points
          cv::Point point(col, row);
          points.push_back(point);
          // fill region with color, ensure correctness of point
          dst.at<Vec3b>(point.y, point.x) = color;
        }
      }
    }
    // save componets point set
    areaPointsets.push_back({area, points});
    // regionColors.push_back(color);
    // selectedLabels.push_back(label);
  }

  // Sort in descending order based on the float value
  std::sort(areaPointsets.begin(), areaPointsets.end(), [](const std::pair<int, std::vector<cv::Point>> &a, const std::pair<int, std::vector<cv::Point>> &b)
            { return a.first > b.first; });

  int count = 0;

  // save N most large area
  for (const auto &points : areaPointsets)
  {
    if (count < maxComponets)
    {
      Pointsets.push_back(points.second);
      count += 1;
    }
    else
    {
      break;
    }
  }

  areaPointsets.clear();

  return (Pointsets.size());
}

/**
 * @brief
 * draw minimum area rectangle,
 * show angle
 * invariant to scale, translation, and rotation
 * @param src
 * @param dst
 * @param Pointsets
 * @return int
 */
int leastInertiaInvariance(cv::Mat &src, cv::Mat &dst, std::vector<std::vector<cv::Point>> &Pointsets, vector<vector<float>> &featureMap)
{
  // display color minRect area
  cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);

  // calculate orientation of least inertia axis and bounding box
  // TODO: percent fill
  // minAreaRect needs at least 5 points to work

  for (size_t i = 0; i < Pointsets.size(); ++i)
  {
    cv::RotatedRect minRect = cv::minAreaRect(Pointsets[i]);

    // sum should be high to ensure the color is bright enough.
    cv::Scalar color = colorList[i];

    // Draw the minimum area rectangle
    cv::Point2f rect_points[4];
    minRect.points(rect_points);
    for (int j = 0; j < 4; j++)
    {
      cv::line(dst, rect_points[j], rect_points[(j + 1) % 4], color);
    }

    // minRect contains: center position, dimensions (width, height), and the rotation angle
    cv::Point2f center = minRect.center;
    cv::Size2f dimensions = minRect.size;

    // percent fills (Area of Each Region /rotated minimum AreaBox)
    // Scale and Translate and rotation Invariance
    float percentfill = Pointsets[i].size() / (dimensions.width * dimensions.height) * 100;

    // angle between horizontal axis and rectangle's longer side (width).
    // [−90,0), 0 indicates that the longer side is horizontal, -90 indicates vertical.
    // angle is invariance to translation and scale
    // float angle = minRect.angle;

    // WHratios is invariance to scale, rotation and translation
    float absWHratios = std::max(dimensions.width, dimensions.height) / std::min(dimensions.width, dimensions.height);

    std::ostringstream text;
    text.precision(3);
    // text << "Angle: " << std::fixed << angle;
    // cv::putText(dst, text.str(), center, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
    text.str("");
    text << "Ratio: " << std::fixed << absWHratios;
    cv::putText(dst, text.str(), Point2f(center.x, center.y + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, color);

    // printf("centroid of least moment: (%.2lf, %.2lf)\n", center.x, center.y);
    // printf("Width Height ratios: %.2lf\n", absWHratios);
    // printf("Orientation of rotation: %.2lf\n", angle);

    featureMap[i].push_back(absWHratios);
    featureMap[i].push_back(percentfill);
    // featureMap[i].push_back(angle);
  }
  if (!featureMap.empty())
  {
    return featureMap[0].size();
  }
  return 0;
}

/**
 * @brief
 * compute hu moments
 * @param Pointsets
 * @param featureMap
 * @return int
 */
int MomentInvariance(std::vector<std::vector<cv::Point>> &Pointsets, vector<vector<float>> &featureMap)
{
  // The first 6 Hu Moments are invariant to translation, scale, and rotation
  // The 7th will change sign based on rotation angle, but invariant to rotation magnitude
  int momentNum = 6;
  for (size_t i = 0; i < Pointsets.size(); ++i)
  {
    cv::Moments moments = cv::moments(Pointsets[i]);
    double huMoments[7];
    cv::HuMoments(moments, huMoments);

    // just keep first 6 humoments
    for (int m = 0; m < momentNum; ++m)
    {
      // logarithmic transformation to deal with wide range of values
      // add small value to avoid log(0)
      huMoments[m] = -1 * copysign(1.0, huMoments[m]) * log10(abs(huMoments[m]) + 1e-7);
      featureMap[i].push_back(static_cast<float>(huMoments[m]));
    }
  }

  return momentNum;
}

/**
 * @brief
 * extract all features from object Mat
 * @param src
 * @param featureVec
 * @param featureSize
 * @return int featureVec dimension, 0 means fail to extract
 */
int extractFeatureVector(cv::Mat &src, std::vector<float> &featureVec, FeatureSize featureSize)
{

  int componentNum = 0;
  float orientation, absWHratios;
  cv::Mat thres, cleanup, segmentRegion, leastInertia;
  std::vector<std::vector<cv::Point>> Pointsets;

  dynamicThreshold(src, thres);
  imshow("threshold", thres);
  cleanupBinary(thres, cleanup);
  imshow("cleanup", cleanup);

  // first feature is the amount of unconnected components
  componentNum = segmentIntoRegion(cleanup, segmentRegion, Pointsets);

  // check if successfully find region
  if(componentNum == 0){
    return 0;
  }

  vector<vector<float>> varableSizefeatureMap(componentNum);

  int feaVecDimension = 0;
  // extract WHratios, orientation, percentfill
  feaVecDimension += leastInertiaInvariance(cleanup, leastInertia, Pointsets, varableSizefeatureMap);
  // extract moments
  feaVecDimension += MomentInvariance(Pointsets, varableSizefeatureMap);

  featureVec.clear();

  if (featureSize == FeatureSize::fixedSize)
  {
    featureVec.resize(1 + feaVecDimension);
  }
  else
  {
    featureVec.resize(1);
  }

  featureVec[0] = static_cast<float>(componentNum);

  // for fixed size feature, use Feature Aggregation, calculate sum of each feature across all components
  for (int i = 0; i < componentNum; ++i)
  {
    for (int j = 0; j < feaVecDimension; ++j)
    {
      float featureVal = varableSizefeatureMap[i][j];
      if (featureSize == FeatureSize::fixedSize)
      {
        featureVec[1 + j] += featureVal;
      }
      else
      {
        featureVec.push_back(featureVal);
      }
      // print feature vector for test
      std::cout << std::fixed << std::setprecision(6) << featureVal << " ";
    }
    std::cout << "\n";
  }
  std::cout << endl;
  varableSizefeatureMap.clear();

  imshow("segment Region", segmentRegion);
  imshow("least Inertia", leastInertia);

  return featureVec.size();
}

/**
 * @brief
 * find closet label
 * @param distances
 * @param labels
 * @return std::string
 */
std::string findNearestNeighbor(const std::vector<double> &distances, const std::vector<char *> &labels)
{
  double minDistance = std::numeric_limits<double>::max();
  std::string nearestLabel = unknownClass;
  for (size_t i = 0; i < distances.size(); ++i)
  {
    if (distances[i] < minDistance)
    {
      minDistance = distances[i];
      nearestLabel = labels[i];
    }
  }
  return nearestLabel;
}

/**
 * @brief
 * find closet label using KNN and Mean Of Distances per class method
 * @param distances
 * @param labels
 * @return std::string
 */
std::string findKNN_meanOfDistance(const std::vector<double> &distances, const std::vector<char *> &labels, int K = 3)
{
  std::vector<std::pair<double, std::string>> distanceLabelPairs;
  std::string label;
  // Combine distances and labels
  for (size_t i = 0; i < distances.size(); ++i)
  {
    distanceLabelPairs.push_back(std::make_pair(distances[i], std::string(labels[i])));
  }

  // 1. Sort distance descending
  std::sort(distanceLabelPairs.begin(), distanceLabelPairs.end());

  // 2 sum distances and count for each class among the
  printf("mean of distances");
  std::unordered_map<std::string, std::pair<double, int>> sumCountperClass;
  for (int i = 0; i < K; ++i)
  {
    std::string label = distanceLabelPairs[i].second;
    double distance = distanceLabelPairs[i].first;
    sumCountperClass[label].first += distance;
    sumCountperClass[label].second += 1;
  }

  // print distance and label for test
  // default is reset
  std::ofstream out("distances.md", std::ios::app);

  // 3 calculating mean distance per class and find the class with the minimum distance
  double minDistance = std::numeric_limits<double>::max();
  std::string nearestLabel = unknownClass;
  for (const auto &entry : sumCountperClass)
  {
    const std::string &label = entry.first;
    double sumDistancePerclass = entry.second.first;
    int count = entry.second.second;
    double meanDistancePerClass = sumDistancePerclass / count;
    if (meanDistancePerClass < minDistance)
    {
      minDistance = meanDistancePerClass;
      nearestLabel = label;
    }

    // print distance and label for test
    out << "mean of distances K=" << K << "\n";
    out << " " << label << " |";
    out << " " << std::fixed << std::setprecision(16) << meanDistancePerClass;
    out << "\n";
  }

  out << "\n";
  // Close the file
  out.close();

  return nearestLabel;
}

/**
 * @brief
 * find closet label using KNN and Sum Of Distances per class method
 * @param distances
 * @param labels
 * @return std::string
 */
std::string findKNN_sumOfDistance(const std::vector<double> &distances, const std::vector<char *> &labels, int K = 3)
{

  std::vector<std::pair<double, std::string>> distanceLabelPairs;
  std::string label;
  // Combine distances and labels
  for (size_t i = 0; i < distances.size(); ++i)
  {
    distanceLabelPairs.push_back(std::make_pair(distances[i], std::string(labels[i])));
  }

  // Sort distance descending
  std::sort(distanceLabelPairs.begin(), distanceLabelPairs.end());

  // sum of distances for each class among the K
  std::unordered_map<std::string, double> eachclassDistance;
  for (int i = 0; i < K; ++i)
  {
    std::string label = distanceLabelPairs[i].second;
    double distance = distanceLabelPairs[i].first;
    eachclassDistance[label] += distance;
  }

  // Find the class with the minimum sum of distances
  double minDistanceSum = std::numeric_limits<double>::max();
  std::string nearestLabel = unknownClass;
  for (const auto &pair : eachclassDistance)
  {
    if (pair.second < minDistanceSum)
    {
      minDistanceSum = pair.second;
      nearestLabel = pair.first;
    }
  }

  // print distance and label for test
  // default is reset
  std::ofstream out("distances.md", std::ios::app);
  // Print the header row for labels
  out << "sum of distances K=" << K << "\n";
  for (const auto &pair : distanceLabelPairs)
  {
    out << " " << pair.second << " |";
    out << " " << std::fixed << std::setprecision(16) << pair.first;
    out << "\n";
  }
  out << "\n";
  // Close the file
  out.close();

  return nearestLabel;
}

/**
 * @brief
 * find closet label using KNN and weighted voting per class method
 * closer label with max voting weight
 * @param distances
 * @param labels
 * @param K
 * @return std::string
 */
std::string findKNN_weightedVoting(const std::vector<double> &distances, const std::vector<char *> &labels, int K = 3)
{

  std::vector<std::pair<double, std::string>> distanceLabelPairs;
  std::string label;
  // Combine distances and labels
  for (size_t i = 0; i < distances.size(); ++i)
  {
    distanceLabelPairs.push_back(std::make_pair(distances[i], std::string(labels[i])));
  }

  // Sort distance descending
  std::sort(distanceLabelPairs.begin(), distanceLabelPairs.end());

  std::unordered_map<std::string, double> eachclassWeight;

  // weighted voting for each class among the K
  for (int i = 0; i < K; ++i)
  {
    std::string label = distanceLabelPairs[i].second;
    double distance = distanceLabelPairs[i].first;
    // avoid division by zero and ensure minimum distances give higher weights
    double weight = 1.0 / (distance + std::numeric_limits<double>::epsilon());

    // sum weights for each class
    eachclassWeight[label] += weight;
  }

  // Find the class with the minimum sum of distances
  double maxWeight = std::numeric_limits<double>::min();
  std::string nearestLabel = unknownClass;
  for (const auto &pair : eachclassWeight)
  {
    if (pair.second > maxWeight)
    {
      maxWeight = pair.second;
      nearestLabel = pair.first;
    }
  }

  // print distance and label for test
  // default is reset
  std::ofstream out("distances.md", std::ios::app);
  // Print the header row for labels
  out << "Weighed Voting K=" << K << "\n";
  for (const auto &pair : distanceLabelPairs)
  {
    out << " " << pair.second << " |";
    out << " " << std::fixed << std::setprecision(16) << pair.first;
    out << "\n";
  }
  out << "\n";
  // Close the file
  out.close();

  return nearestLabel;
}

/**
 * @brief
 * classify new image, return label
 * using given object database
 * using Given distance metric (Euclidean distance,cosine distance,squared difference )
 * using Given Classification method(Nearest Neighbor matching, K Nearest Neighbor)
 * @param unknown
 * @param featureVec
 * @return label, fail info for extraction
 */
std::string classify(cv::Mat &unknown, const char *objectDBfilename, DistanceMetric metric, ClassificationMethod method)
{

  std::string label;
  vector<float> unknownFea;
  std::vector<double> distances;
  std::vector<char *> labels;

  // calculate unknown feature
  int featureVecsize = extractFeatureVector(unknown, unknownFea, FeatureSize::variableSize);
  // fail to extract feature, unkown
  if(featureVecsize==0){
    return("fail to extract features");
  }

  
  // calculate distances and labels using Distance Metric
  distances_from_csv(unknownFea, objectDBfilename, distances, labels, metric);

  // find closet label using Classification Method
  switch (method)
  {
  case ClassificationMethod::NN:
    label = findNearestNeighbor(distances, labels);
    break;
  case ClassificationMethod::KNN:
    label = findKNN_meanOfDistance(distances, labels, 3);
    // label = findKNN_sumOfDistance(distances, labels, 3);
    // label = findKNN_weightedVoting(distances, labels, 3);
    break;
  default:
    throw std::invalid_argument("unsupported Classification Method distance metric");
  }

  // TODO: update DB with new label
  if (label == unknownClass)
  {
  }

  return label;
}