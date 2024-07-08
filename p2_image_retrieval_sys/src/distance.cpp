/*
  Author: Yilin Tang
  Date: 2024-02-10
  CS 5330 Computer Vision
  Description: 

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

using namespace cv;

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

double cosinedistance(const std::vector<float> &feaVec1, const std::vector<float> &feaVec2)
{
  if (feaVec1.size() != feaVec2.size())
  {
    throw std::invalid_argument("feature vec must be of the same size.");
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
 * calculate all distance for single histograms intersections for multiple hists
 */
int cosinedistances_from_csv(char *targetname, char *csvfilename, std::vector<double> &distances,
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
int custom_DNN_distances_from_csv(char *targetname, std::vector<std::vector<float>> &targetFeas, std::vector<char *> &csvfiles,
                                  std::vector<float> &weights, std::map<double, std::string> &distancesmap)
{

  std::vector<double> distances, HistInts;
  std::vector<char *> filenames;

  // the first feature is DNN, not weight yet
  cosinedistances_from_csv(targetname, csvfiles[0], distances, filenames, 1.0);
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