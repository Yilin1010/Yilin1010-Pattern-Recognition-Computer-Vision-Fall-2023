/*
  Author: Yilin Tang
  Date: 2024-01-12
  CS 5330 Computer Vision
  Description: 

    display image
    apply multiple filters to selected image from video
    
    usage: imgDisplay.out <imgs_Path>
    compile: -o executable vidDisplay.cpp filter.cpp
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "filter.h"
using namespace cv;

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


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread(argv[1], 1);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);

    Mat f1,f2;
    blur5x5_1(image, f1);
    imshow("f1", f1);
    blur5x5_2(image, f2);
    imshow("f2", f2);

    Mat f1,f2;
    sepiatone1(image, f1);
    imshow("f1", f1);
    sepiatone2(image, f2);
    imshow("f2", f2);
    

    Mat x,y,absX,absY;
    sobelX3x3(image, x);
    sobelY3x3(image, y);

    // convert [-255,255] to [0,255]
    convertScaleAbs(x,absX);
    convertScaleAbs(y,absY);
    // visualize the Sobel outputs as separate variables.
    imshow("sobelX", absX);
    imshow("sobelY", absY);


    // 'm' key shows the color gradient magnitude image.
    Mat gradientImg;
    magnitude(absX,absY,gradientImg);
    imshow("gradient", absX);


    // 'the 'l' key displays 'blurQuantize image.
    Mat blurQuantizedImg;
    blurQuantize(image, blurQuantizedImg, 10);
    imshow("blurQuantize", blurQuantizedImg);

    Mat cartoonized;
    cartoonize(image, cartoonized);
    imshow("cartoonize", cartoonized);

    cv::Mat luminanceQuantized;
    luminanceQuantization(image,luminanceQuantized,4);
    imshow("luminance Quantized", luminanceQuantized);

    cv::Mat edge;
    edgeDetection(image,edge);
    imshow("edge", edge);

    cv::Mat cartoonized;
    cartoonize(image,cartoonized);
    imshow("cartoonize", cartoonized);

    waitKey(0);
    return(0);
}