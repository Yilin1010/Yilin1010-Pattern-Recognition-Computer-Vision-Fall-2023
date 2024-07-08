/*
  Author: Yilin Tang
  Date: 2024-02-10
  CS 5330 Computer Vision
  Description: 

*/

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int histToFeatureVector(const cv::Mat &hist, std::vector<float> &featureVector)
{
  // Ensure the input histogram is not empty.
  if (hist.empty())
  {
    throw std::invalid_argument("Empty histogram.");
  }
  // flatten the histogram matrix into a one-dimensional array.

  // Reserve space for efficiency.
  featureVector.reserve(hist.rows * hist.cols);

  for (int r = 0; r < hist.rows; ++r)
  {
    for (int c = 0; c < hist.cols; ++c)
    {
      float value = hist.at<float>(r, c);
      featureVector.push_back(value);
    }
  }

  return 0;
}

double sevenSquare(const Mat &target, const Mat &compare)
{

  int startr = (target.rows - 7) / 2; // 512 252+7=259 255
  int startc = (target.cols - 7) / 2;

  Rect roi(startr, startc, 7, 7);
  Mat Tfea = target(roi);
  Mat Cfea = compare(roi);

  Mat diff;
  cv::absdiff(Tfea, Cfea, diff);
  diff = diff.mul(diff);
  cv::Scalar sumsofCh = cv::sum(diff);

  double ssd = 0;
  for (int i = 0; i < diff.channels(); ++i)
  {
    ssd += sumsofCh[i];
  }

  return ssd;
}

int rgbHistfeature(const Mat &image, std::vector<float> &featureVector, int bins)
{

  // Clear to deallocate memory, otherwise give wrong distances
  featureVector.clear();
  // initialize the histogram (use floats so we can make probabilities)
  featureVector.resize(bins * bins * bins, 0);

  for (int r = 0; r < image.rows; ++r)
  {
    for (int c = 0; c < image.cols; ++c)
    {
      cv::Vec3b pixel = image.at<cv::Vec3b>(r, c);

      float blue = pixel[0];
      float green = pixel[1];
      float red = pixel[2];

      // compute r, g and Normalize them to [0,1]
      float divisor = red + green + blue;
      divisor = divisor > 0.0 ? divisor : 1.0; // check for all zeros

      float rvalue = red / divisor;
      float gvalue = green / divisor;
      float bvalue = blue / divisor;

      // compute index of bin, for exp: bins = 10, 0~0.1 index = 0
      int rindex = (int)(rvalue * (bins - 1) + 0.5);
      int gindex = (int)(gvalue * (bins - 1) + 0.5);
      int bindex = (int)(bvalue * (bins - 1) + 0.5);

      featureVector[rindex * (bins * bins) + gindex * bins + bindex] += 1.0; // Increment histogram bin.
    }
  }
  return 0;
}

/**
 * @brief 
 * caculate green and blue channel
 * @param image 
 * @param featureVector 
 * @param bins 
 * @return int 
 */
int gbHistfeature(const Mat &image, std::vector<float> &featureVector, int bins)
{

  // Clear to deallocate memory, otherwise give wrong distances
  featureVector.clear();
  // initialize the histogram (use floats so we can make probabilities)
  featureVector.resize(bins * bins, 0);

  for (int r = 0; r < image.rows; ++r)
  {
    for (int c = 0; c < image.cols; ++c)
    {
      cv::Vec3b pixel = image.at<cv::Vec3b>(r, c);

      float blue = pixel[0];
      float green = pixel[1];
      float red = pixel[2];

      // compute r, g and Normalize them to [0,1]
      float divisor = red + green + blue;
      divisor = divisor > 0.0 ? divisor : 1.0; // check for all zeros

      float gvalue = green / divisor;
      float bvalue = blue / divisor;

      // compute index of bin, for exp: bins = 10, 0~0.1 index = 0
      int gindex = (int)(gvalue * (bins - 1) + 0.5);
      int bindex = (int)(bvalue * (bins - 1) + 0.5);

      featureVector[gindex * bins + bindex] += 1.0; // Increment histogram bin.
    }
  }
  return 0;
}

/**
 * @brief
 * calculate feature representing the center of image
 * @param image
 * @param featureVector
 * @param bins
 * @return * feature
 */
int centerHistfeature(const Mat &image, std::vector<float> &featureVector, int bins)
{

  // clip the center quarter
  int startr = image.rows / 3;
  int startc = image.cols / 3;

  Rect roi(startc, startr, image.rows / 3, image.cols / 3);
  Mat center = image(roi);

  rgbHistfeature(center, featureVector, bins);

  return 0;
}

/**
 * @brief
 * feature represents the 2/3 top of image
 * @param image
 * @param featureVector
 * @param bins
 * @return * int
 */
int topHistfeature(const Mat &image, std::vector<float> &featureVector, int bins)
{

  // clip the center quarter
  Rect roi(0, 0, image.cols, (image.rows * 2) / 3);
  Mat top = image(roi);

  rgbHistfeature(top, featureVector, bins);

  return 0;
}

/**
 * @brief
 * caculate orientation for each pixel given sobel,
 * to be used as feature for 2d histogram of oritentaion and magnitude
 * @param sobelX
 * @param sobelY
 * @param orientation
 */
void calOrientation(const cv::Mat &sobelX, const cv::Mat &sobelY, cv::Mat &orientation)
{
  // output should have same size with input sobel
  orientation.create(sobelX.rows, sobelX.cols, CV_32FC3);

  // Set 'angleInDegrees' to true to get results in degrees, [0,360)
  cv::phase(sobelX, sobelY, orientation, true);
}

/**
 * @brief
 * feature extract from 2d histogram of oritentaion and magnitude
 * @param image
 * @param featureVector
 * @param bins
 * @return * int
 */
int textureHistfeature(Mat &image, std::vector<float> &featureVector)
{

  int magnitudeBins = 16;
  int orientationBins = 18; // 20*18 = 360

  // Clear to deallocate memory, otherwise give wrong distances
  featureVector.clear();
  // initialize the histogram (use floats so we can make probabilities)
  featureVector.resize(orientationBins * magnitudeBins, 0);

  // calculate magnitude and orientation
  Mat grayImage, x, y, absX, absY;
  cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
  cv::Sobel(grayImage, x, CV_32F, 1, 0);
  cv::Sobel(grayImage, y, CV_32F, 1, 0);

  // convert [-255,255] to [0,255]
  // convertScaleAbs(x, absX);
  // convertScaleAbs(y, absY);

  Mat mag, ori;
  // magnitude(absX, absY, mag);
  // calOrientation(x, y, ori);
  cv::cartToPolar(x, y, mag, ori, true);
  cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);

  // Replace NaNs with 0 or a suitable value
  // cv::patchNaNs(ori, 0);

  for (int r = 0; r < image.rows; ++r)
  {
    for (int c = 0; c < image.cols; ++c)
    {
      float magValue = mag.at<float>(r, c);
      float oriValue = ori.at<float>(r, c);

      // Normalize magnitude and orientation and caculate index of bins

      int magIndex = (int)((magValue / 255) * (magnitudeBins - 1) + 0.5);
      int oriIndex = (int)((oriValue / 360) * (orientationBins - 1) + 0.5);

      featureVector[magIndex * magnitudeBins + oriIndex] += 1.0; // Increment histogram bin.
    }
  }
  return(0);
}

