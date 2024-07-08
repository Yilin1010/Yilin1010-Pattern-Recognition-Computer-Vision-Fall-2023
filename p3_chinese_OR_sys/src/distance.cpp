/*
  Author: Yilin Tang
  Date: 2024-02-10
  CS 5330 Computer Vision
  Description:

  support histogram intersection, scaled Euclidean distance, Cosine distance
  calculated distances from CSV
  provided weighted distances for multiple features
*/

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <stdio.h>
#include <numeric> //std::accumulate
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <iostream>
#include "csv_util.h"
#include "distance.h"

using namespace cv;
using namespace std;

/**
 * @brief
 * calculate Standard Deviation for each feature(each dimension or column)across database
 * @param features
 * @param stdevs
 * @return double
 */
int StandardDeviation(std::vector<std::vector<float>> &features, std::vector<float> &stdevs)
{
  if (features.empty())
    return 1;

  size_t numFeatures = features[0].size();
  for (const auto &featureVec : features)
  {
    if (featureVec.size() != numFeatures)
    {
      throw std::invalid_argument("All feature vectors must have same size.");
    }
  }

  stdevs.resize(numFeatures, 0.0);

  // Calculate the mean of each feature
  std::vector<float> means(numFeatures, 0.0);
  for (const auto &featureVec : features)
  {
    for (size_t i = 0; i < numFeatures; ++i)
    {
      means[i] += featureVec[i];
    }
  }
  for (float &mean : means)
  {
    mean /= features.size();
  }

  // Calculate standard deviation
  for (const auto &featureVec : features)
  {
    for (size_t i = 0; i < numFeatures; ++i)
    {
      float diff = featureVec[i] - means[i];
      stdevs[i] += diff * diff;
    }
  }
  for (float &stdev : stdevs)
  {
    stdev = std::sqrt(stdev / features.size());
  }

  return 0;
}

/**
 * @brief
 * calculate histogram intersection for two feature
 * @param feaVec1
 * @param feaVec2
 * @return double
 */
double histIntersection(const std::vector<float> &feaVec1, const std::vector<float> &feaVec2)
{
  if (feaVec1.size() != feaVec2.size())
  {
    throw std::invalid_argument("feature vec must be of the same size.");
  }

  double intersection = 0.0;
  for (size_t i = 0; i < feaVec1.size(); ++i)
  {
    intersection += std::min(feaVec1[i], feaVec2[i]);
  }

  return intersection;
}

/**
 * @brief
 * cosine distance range, 0 indentical, 2 most different, -1 different feature vector size
 * @param feaVec1
 * @param feaVec2
 * @return double
 */
double cosinedistance(const std::vector<float> &feaVec1, const std::vector<float> &feaVec2)
{
  if (feaVec1.size() != feaVec2.size())
  {
    // encode different feature vector size as most different  
    return (-1.0);
  }

  double sumOfProduct = std::inner_product(feaVec1.begin(), feaVec1.end(), feaVec2.begin(), 0.0);
  // L2 norm
  double sumofMagnitude1 = std::inner_product(feaVec1.begin(), feaVec1.end(), feaVec1.begin(), 0.0);
  double sumofMagnitude2 = std::inner_product(feaVec2.begin(), feaVec2.end(), feaVec2.begin(), 0.0);

  double cosinedisntance = 1 - sumOfProduct / (std::sqrt(sumofMagnitude1) * std::sqrt(sumofMagnitude2));

  return cosinedisntance;
}

/**
 * @brief
 * standardization: scaled Euclidean distance by the standard deviation
 * each feature with different scales contributes equally to the distance
 * 0 to infinity, 0 indicates indentical, -1 indicates different feature vector size
 * @param feaVec1
 * @param feaVec2
 * @return double
 */
double scaledEuclideandistance(const std::vector<float> &feaVec1, const std::vector<float> &feaVec2, const std::vector<float> &stdevs)
{
  if (feaVec1.size() != feaVec2.size())
  {
    return(-1.0);
    // throw std::invalid_argument("feature vec must be of the same size.");
  }
  double sum = 0;
  for (size_t i = 0; i < feaVec1.size(); ++i)
  {
    if (stdevs[i] != 0)
    { // avoid division by zero
      double diff = (feaVec1[i] - feaVec2[i]) / stdevs[i];
      sum += diff * diff;
    }   

  }
  return std::sqrt(sum);
}

/**
 * @brief
 * calculate Euclidean distance, cosine distance given feature vector database
 * return resulting distances and corresponding labels vector
 * if the feature vector dimension is not same, the distance won't be computed and that label will be removed 
 * @param targetFea given new feature vector
 * @param csvfilename
 * @param distances output 
 * @param labels output 
 * @param metric default DistanceMetric::EuclideanDistance
 * @param weight default 0
 * @return int 
 */
int distances_from_csv(std::vector<float> &targetFea, const char *csvfilename, std::vector<double> &distances, std::vector<char *> &labels, DistanceMetric metric, float weight)
{

  // read data from csv
  std::vector<std::vector<float>> features;
  read_image_data_csv(csvfilename, labels, features, 0);

  
  int dataset_num = labels.size();
  distances.clear();
  std::vector<char *> removedlabels; 

  for (std::size_t i = 0; i < features.size(); ++i)
  {
    const auto &label = labels[i];
    const auto &feaVecC = features[i];

    double distance;
    switch (metric)
    {
    case DistanceMetric::EuclideanDistance:
    {
      std::vector<float> stdevs;
      StandardDeviation(features, stdevs);
      distance = scaledEuclideandistance(targetFea, feaVecC, stdevs);
      break;
    }
    case DistanceMetric::CosineDistance:
      distance = cosinedistance(targetFea, feaVecC);
      break;
    default:
      throw std::invalid_argument("Unsupported distance metric");
    }

    // remove distance and label with different feature vector size
    // "close enough" to -1 
    // using a small epsilon value to allow for floating-point arithmetic inaccuracies
    if (std::fabs(distance + 1) < 1e-9){
      continue;
    }

    removedlabels.push_back(labels[i]);
    distances.push_back(distance * weight);
  }

  printf("%d remains %d after removing feature vectors with different size\n",dataset_num,(int)removedlabels.size());
  labels.clear();
  labels = removedlabels;
  return (distances.size());
}

/**
 * @brief 
 * calculate all distance for single histograms intersections for multiple hists
 * @param targetname 
 * @param csvfilename 
 * @param distances 
 * @param filenames 
 * @param weight 
 * @return int 
 */
int cosinedistances_from_csv_byName(char *targetname, char *csvfilename, std::vector<double> &distances,
                                    std::vector<char *> &filenames, float weight = 1.0)
{
  // read data from csv
  std::vector<std::vector<float>> features;
  read_image_data_csv(csvfilename, filenames, features, 0);

  // get target feature by filename
  std::vector<float> targetFea;
  auto iter = std::find_if(filenames.begin(), filenames.end(), [&targetname](const char *str)
                           { return std::strcmp(str, targetname) == 0; });
  size_t targetIndex = std::distance(filenames.begin(), iter);
  targetFea = features[targetIndex];

  // resize distances array
  if (distances.empty())
  {
    distances.resize(features.size(), 0);
  }

  for (std::size_t i = 0; i < features.size(); ++i)
  {
    const auto &filename = filenames[i];
    const auto &feaVecC = features[i];

    double consinedistance = cosinedistance(targetFea, feaVecC);

    distances[i] += consinedistance * weight;
  }

  return (features.size());
}

/**
 * @brief
 * calculate all distance for single histograms intersections for multiple hists
 */
int histIntDistances_from_csv(std::vector<float> &targetFea, char *csvfilename, std::vector<double> &distances,
                              std::vector<char *> &filenames, float weight = 1.0)
{

  // read data from csv
  std::vector<std::vector<float>> features;
  // clear filenames of previous features
  int previousFeaSize = filenames.size();
  filenames.clear();

  read_image_data_csv(csvfilename, filenames, features, 0);

  if ((previousFeaSize != 0 && previousFeaSize != features.size()) ||
      (!distances.empty() && distances.size() != features.size()))
  {
    throw std::invalid_argument("distances, features and filenames should have same number of feature");
  }

  // resize distances array
  if (distances.empty())
  {
    distances.resize(features.size(), 0);
  }

  // get sum of Target hist
  float sumHistT = std::accumulate(targetFea.begin(), targetFea.end(), 0);

  for (std::size_t i = 0; i < features.size(); ++i)
  {
    const auto &filename = filenames[i];
    const auto &feaVecC = features[i];

    double featuredistance = histIntersection(targetFea, feaVecC);

    double distance = featuredistance / sumHistT;

    // add weighted distance to existing distance, if it is single hist it will be 0 + distance
    distances[i] += distance * weight;
  }

  return (features.size());
}
/**
 * @brief
 * calculate distance for weighted histograms intersections for multiple hists
 * corresponding feature elements must be same order in array
 * @param targetFea1
 * @param targetFea2
 * @param csvfile
 * @param weight
 * @return int
 */
int mulhist_distances_from_csv(std::vector<std::vector<float>> &targetFeas, std::vector<char *> &csvfiles,
                               std::vector<float> &weights, std::map<double, std::string> &distancesmap)
{

  if (targetFeas.size() != csvfiles.size() || weights.size() != csvfiles.size() && targetFeas.size() != weights.size())
  {
    throw std::invalid_argument("csvfiles, targetFeas and weights must be same size");
  }

  std::vector<double> distances;
  std::vector<char *> filenames;

  for (int i = 0; i < csvfiles.size(); ++i)
  {

    histIntDistances_from_csv(targetFeas[i], csvfiles[i], distances, filenames, weights[i]);

    printf("computed %d distances\n, weight: %.2lf\n", (int)filenames.size(), weights[i]);
  }

  for (int i = 0; i < filenames.size(); ++i)
  {
    distancesmap.insert(std::make_pair(distances[i], (std::string)(filenames[i])));
  }

  return (0);
}

/**
 * @brief
 * calculate single DNN distances or multiple feature distance inclued DNN
 * @param targetname
 * @param targetFeas
 * @param csvfiles
 * @param weights
 * @param distancesmap
 * @return int
 */
int custom_DNN_distances_from_csv_byName(char *targetname, std::vector<std::vector<float>> &targetFeas, std::vector<char *> &csvfiles,
                                         std::vector<float> &weights, std::map<double, std::string> &distancesmap)
{

  std::vector<double> distances, HistInts;
  std::vector<char *> filenames;

  // the first feature is DNN, not weight yet
  cosinedistances_from_csv_byName(targetname, csvfiles[0], distances, filenames, 1.0);
  printf("computed %d cosine distances\n, weight: %.2lf\n", (int)filenames.size(), weights[0]);

  HistInts.resize(filenames.size(), 0);
  // add weighted second feature if it is given
  for (int i = 1; i < csvfiles.size(); ++i)
  {

    histIntDistances_from_csv(targetFeas[i - 1], csvfiles[i], HistInts, filenames, weights[i]);

    printf("computed %d hist Inte distances\n, weight: %.2lf\n", (int)filenames.size(), weights[i]);
  }

  if (targetFeas.empty())
  {
    for (int i = 0; i < filenames.size(); ++i)
    {
      distancesmap.insert(std::make_pair(weights[0] * distances[i], (std::string)(filenames[i])));
    }
  }
  else
  {
    for (int i = 0; i < filenames.size(); ++i)
    {
      // Normalize the Measures, higher value indicates greater similarity
      distancesmap.insert(std::make_pair(weights[0] * (1 - distances[i]) + HistInts[i], (std::string)(filenames[i])));
    }
  }

  return (0);
}