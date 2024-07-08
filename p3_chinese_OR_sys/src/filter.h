/*
  Author: Yilin Tang
  Date: 2024-01-25
  CS 5330 Computer Vision
  Description: 

  Include file for vidDisplay.cpp, filter functions
*/

#ifndef FILTER_H
#define FILTER_H

// prototypes

int greyscale(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sepiatone1(cv::Mat &src, cv::Mat &dst);
int sepiatone2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );
int adaptiveThresholdMeanBinary(cv::Mat &src, cv::Mat &dst, int blocksize, float constant);
int luminanceQuantization(cv::Mat &src, cv::Mat &dst,int K_levels);
int edgeDetection(cv::Mat &src, cv::Mat &dst);
int cartoonize(cv::Mat &src, cv::Mat &dst);
int cartoonizeface( cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces, int minWidth = 50);



#endif // FILTER_H