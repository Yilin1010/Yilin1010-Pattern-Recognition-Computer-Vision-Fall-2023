/*
  Author: Yilin Tang
  Date: 2024-01-25
  CS 5330 Computer Vision
  Description: 

  1.implement multiuple image transformation algorithms, include
    color space conversion 
    filter
    blur
    adaptive threshold
    luminance quantization & color quantization
    edge detector
    cartoonization
  2. implement parallel pixel-wise computation.
*/

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "faceDetect.h"
#include "filter.h"
using namespace cv;
using namespace std;

int luminanceQuantization(cv::Mat &src, cv::Mat &dst, int);
int edgeDetection(cv::Mat &src, cv::Mat &dst);
int adaptiveThresholdMeanBinary(cv::Mat &src, cv::Mat &dst, int blocksize, float constant);

/**
 * @brief 
 * clip each value in vec3f by max
 * @param vector 
 * @param max 
 * @return cv::Vec3f 
 */
cv::Vec3f clipVec(const cv::Vec3f &vector, float max)
{
    return cv::Vec3f(std::min(vector[0], max),
                     std::min(vector[1], max),
                     std::min(vector[2], max));
}


/**
 * @brief apply Pixel-Wise multiplication Parallelly
 * 
 */
class ParallelPixelMultiply : public cv::ParallelLoopBody
{

public:
    ParallelPixelMultiply(cv::Mat &src_, cv::Mat &dst_, const cv::Matx33f &mat_) : src(src_), dst(dst_), mat(mat_) {}

    virtual void operator()(const cv::Range &range) const CV_OVERRIDE
    {
        for (int r = range.start; r < range.end; r++)
        {
            for (int c = 0; c < src.cols; c++)
            {
                const cv::Vec3b pixel = src.at<cv::Vec3b>(r, c);
                cv::Vec3f floatPixel = cv::Vec3f(pixel[0], pixel[1], pixel[2]);
                cv::Vec3f result = mat * floatPixel;
                result = clipVec(result, 255.0f);
                dst.at<cv::Vec3b>(r, c) = cv::Vec3b(result[0], result[1], result[2]);
                // dst.at<cv::Vec3b>(r, c) =  mat * floatPixel;

            }
        }
    }

private:
    cv::Mat &src;
    cv::Mat &dst;
    cv::Matx33f mat;
};

/**
 * @brief 
 * helper method to apply convolution operation for different filters
 * @param src 
 * @param dst 
 * @param kernel 
 * @return int 0 means success
 */
int convolution1(cv::Mat &src, cv::Mat &dst, cv::Mat &kernel){

    int radius = kernel.rows/2;
    for (int row = 0; row < src.rows-kernel.rows; ++row){
        for (int col = 0; col < src.cols-kernel.cols; ++col) {
            for (int ch = 0; ch < 3; ++ch){
                short centerRes = 0;
                for (int krow = 0; krow < kernel.rows; ++krow){
                    for (int kcol = 0; kcol < kernel.cols; ++kcol){
                        centerRes +=  static_cast<short>(src.at<Vec3b>(row+krow,col+kcol)[ch]) * 
                                     kernel.at<short>(krow,kcol);

                    }
                }
                dst.at<Vec3s>(row+radius,col+radius)[ch] = centerRes;

                // debug print
                // if(row == 100 & col == 100 & ch == 1){
                //     short value = dst.at<Vec3s>(100,100)[ch];
                //     printf("actual center value %d\n",value);}
            }
        }
    }
    return 0;

}


/**
 * @brief 
 * create a grayscale image
 * using the Saturation channel from the HSV color space 
 * as the grayscale tone is determined by the color saturation
 * convert color image to grayscale
 * @param src 
 * @param dst 
 * @return int 
 */
int greyscale(cv::Mat &src, cv::Mat &dst)
{

    dst.create(src.rows, src.cols, CV_8UC1);
    cv::Mat hsv;
    hsv.create(src.rows, src.cols, CV_8UC3);
    cv::cvtColor(src,hsv,COLOR_BGR2HSV);
    

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = hsv.at<cv::Vec3b>(y, x);
            // Calculate the grayscale value using the luminosity method
            uchar grayValue = static_cast<uchar>(
                pixel[2] * 0.7 +  // value 
                pixel[1] * 0.2 +  // Saturation
                pixel[0] * 0.1);  // Hue
            dst.at<uchar>(y, x) = grayValue;
        }
    }

    // return 0 on success
    return 0;
}


/**
 * @brief 
 * no optimized sepiatone filter
 * @param src 
 * @param dst 
 * @return int 
 */
int sepiatone1(cv::Mat &src, cv::Mat &dst){

    // printf("%s\n", __func__);

    cv::Matx33f mat(0.272, 0.534, 0.131,
                       0.349, 0.686, 0.168,
                       0.393, 0.769, 0.189);
    dst.create(src.rows, src.cols, CV_8UC3);
    for (int r = 0; r < src.rows; r++)
    {
        for (int c = 0; c < src.cols; c++)
        {
            cv::Vec3b pixel = src.at<cv::Vec3b>(r, c);
            cv::Vec3f floatPixel = cv::Vec3f(pixel[0], pixel[1], pixel[2]);
            cv::Vec3f result = mat * floatPixel;
            result = clipVec(result, 255.0f);
            dst.at<cv::Vec3b>(r, c) = cv::Vec3b(result[0], result[1], result[2]);
        }
    }

    // return 0 on success
    return 0;
}

/**
 * @brief 
 * parallel computation across multiple threads,
 * each thread execute pixel-wise multiplication for each row
 * @param src 
 * @param dst 
 * @return int 
 */
int sepiatone2(cv::Mat &src, cv::Mat &dst)
{
    // (blue coefficients column, green , red)
    // blue row * each Vec3b pixel * = blue value
    // [3*3] * [3*1] = [3*1]
    // the Vec3b will treated as 3*1 column vector in Opencv, so it must be right
    // cv::Matx33f matrix(0.189, 0.168, 0.131,
    //                     0.769, 0.686, 0.534,
    //                     0.393, 0.349, 0.272);

    cv::Matx33f matrix(0.272, 0.534, 0.131,
                       0.349, 0.686, 0.168,
                       0.393, 0.769, 0.189);

    dst.create(src.rows, src.cols, CV_8UC3);
    
    ParallelPixelMultiply op(src, dst, matrix);
    cv::parallel_for_(cv::Range(0, src.rows), op);

    // return 0 on success
    return 0;
}

/**
 * @brief 
 * perform no optimized convolution operation
 * handle each color channel separately
 * Use the integer approximation of a Gaussian of 5*5 filter
 * @param src 
 * @param dst same size with src
 * @return int 
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
    cv::Mat kernel = (cv::Mat_<float>(5, 5) << 1, 2, 4, 2, 1, 2, 4, 8, 4, 2, 4, 8, 16, 8, 4, 2, 4, 8, 4, 2, 1, 2, 4, 2, 1);
    
    // Normalize the kernel
    kernel = kernel/(sum(kernel)[0]);

    // Copy the src input image to the dst image
    if(src.type()!=dst.type()){
        dst.create(src.rows,src.cols,CV_8UC3);}
    src.copyTo(dst);

    for (int row = 0; row < src.rows-kernel.rows; ++row){
        for (int col = 0; col < src.cols-kernel.cols; ++col) {
            for (int ch = 0; ch < 3; ++ch){
                float sumofProd = 0;
                for (int krow = 0; krow < kernel.rows; ++krow){
                    for (int kcol = 0; kcol < kernel.cols; ++kcol){
                        sumofProd += src.at<Vec3b>(row+krow,col+kcol)[ch] * 
                                     kernel.at<float>(krow,kcol);
                    }
                }
                // skip outer two rows and columns.
                dst.at<Vec3b>(row+2,col+2)[ch] = std::min(static_cast<int>(sumofProd),255);
            }
        }
    }


    return 0;
}

/**
 * @brief 
 * perform faster convolution operation
 * handle each color channel separately
 * Use the integer approximation of a Gaussian
 * using separable 1x5 filters vertical and horizontal
 * @param src 
 * @param dst 
 * @return int 
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst){
    cv::Mat vecKernel = (cv::Mat_<float>(5,1) << 1, 2, 4, 2, 1);
    cv::Mat horKernel = (cv::Mat_<float>(1,5) << 1, 2, 4, 2, 1);

    // int stores the intermidiate image after vertical filter,
    //  as convolution needs the original value
    cv::Mat intm;
        
    // Copy the src input image to the dst image
    if(src.type()!=dst.type()){
        dst.create(src.rows,src.cols,CV_8UC3);}
    src.copyTo(dst);
    intm.create(src.rows, src.cols, CV_8UC3);

    int ksize = vecKernel.rows;

    // Normalize the kernel
    vecKernel = vecKernel/(sum(vecKernel)[0]);
    horKernel = horKernel/(sum(horKernel)[0]);

    // apply 1*5 filter vertical, sliding up and down each column
    for (int row = 0; row < src.rows - ksize; ++row){
        for (int col = 0; col < src.cols; ++col) {
            for (int ch = 0; ch < 3; ++ch){
                float sumofProd = 0;
                for (int k = 0; k < ksize; ++k){
                    sumofProd += src.at<Vec3b>(row+k,col)[ch] * vecKernel.at<float>(k);
                }
            intm.at<Vec3b>(row+2,col+2)[ch] = std::min(static_cast<int>(sumofProd),255);
            }
        }
    }

    // apply 1*5 filter horizontal to vertical result dst
    //  sliding left to rignt across each row
    for (int row = 0; row < src.rows; ++row){
        for (int col = 0; col < src.cols - ksize; ++col) {
            for (int ch = 0; ch < 3; ++ch){
                short sumofProd = 0;
                for (int k = 0; k < ksize; ++k){
                    sumofProd += intm.at<Vec3b>(row,col+k)[ch] * horKernel.at<float>(k);
                }
            dst.at<Vec3b>(row+2,col+2)[ch] = std::min(static_cast<int>(sumofProd),255);
            }
        }
    }

    // pixel = dst.at<Vec3b>(100,100);
    // printf("pixel %d %d %d\n",pixel[0],pixel[1],pixel[2]);

    // return 0 on success
    return 0;
}

/**
 * @brief 
 *  apply a positive right X Sobel filter
 * @param src  input 3-channel signed short images
 * @param dst  output color images, CV_16SC3
 * @return int 1 success 
 */
int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    dst.create(src.rows, src.cols,CV_16SC3);

    cv::Mat kernel = (cv::Mat_<short>(3, 3) << -1,0,1,-2,0,2,-1,0,1);
    convolution1(src,dst,kernel);

    return 0;
}

/**
 * @brief 
 *  apply a positive up Y Sobel filter
 * @param src  input  3-channel signed short images
 * @param dst  output color images, CV_16SC3
 * @return int 1 success 
 */
int sobelY3x3( cv::Mat &src, cv::Mat &dst ){
    dst.create(src.rows, src.cols,CV_16SC3);

    cv::Mat kernel = (cv::Mat_<short>(3, 3) << 1,2,1,0,0,0,-1,-2,-1);
    convolution1(src,dst,kernel);

    return 0;
}
/**
 * @brief 
 * generates a gradient magnitude image from the X and Y Sobel images
 * using Euclidean distance for magnitude: I = sqrt( sx*sx + sy*sy )
 * @param sx 3-channel signed short images
 * @param sy 3-channel signed short images
 * @param dst a uchar color image
 * @return int 
 */
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){

    dst.create(sx.rows,sx.cols,CV_8UC3);

    for (int row = 0; row < sx.rows; ++row){
        for (int col = 0; col < sx.cols; ++col) {
            for (int ch = 0; ch < 3; ++ch){
                short gx = sx.at<Vec3s>(row,col)[ch];
                short gy = sy.at<Vec3s>(row,col)[ch];
                int distance  = static_cast<int>(std::sqrt(gx*gx + gy*gy));

                dst.at<Vec3b>(row,col)[ch] = std::min(255,distance);
            }
        }
    }

    return 0;
}

/**
 * @brief 
 * 
 * @param src 
 * @param dst 
 * @param levels 
 * @return int 
 */
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){
    // blur5x5_2(src,dst);
    blur5x5_1(src,dst);

     for (int row = 0; row < src.rows; ++row){
        for (int col = 0; col < src.cols; ++col) {
            for (int ch = 0; ch < 3; ++ch){
                int value = src.at<Vec3b>(row,col)[ch];
                dst.at<Vec3b>(row,col)[ch] = static_cast<int>(value/levels)*levels;;
            }
        }
     }
    return 0;
}



/**
 * @brief 
 * quantize the luminance of BGR image by level
 * use k-means to group luminance value
 * converting to YCrCb color space during quantization
 * @param src BGR
 * @param dst BGR
 * @return int 
 */
int luminanceQuantization(cv::Mat &src, cv::Mat &dst, int K_levels){

    cv::Mat YCrCb,Y_channel;
    cv::Mat labels,centroids;

    cvtColor(src, YCrCb,cv::COLOR_BGR2YCrCb);

    YCrCb.convertTo(YCrCb, CV_32F);

    //  Extract the Y (Luma) channel
    cv::extractChannel(YCrCb,Y_channel,0);
    Y_channel.convertTo(Y_channel, CV_32F);


    // # Reshape Y_channel for K-means (channel)
    Y_channel = Y_channel.reshape(1, Y_channel.rows*Y_channel.cols);
    // Y_channel = Y_channel.reshape(1, Y_channel.total());

    // Define criteria and apply kmeans()
    // int K_levels = 4;
    int iter = 30;
    float epsilon = 0.02;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, iter, epsilon);
    cv::kmeans(Y_channel, K_levels, labels, criteria, 10, cv::KMEANS_RANDOM_CENTERS, centroids);
    
    // Reassign Luminance Values
    for (int i = 0; i < Y_channel.rows; ++i) {
        int cluster_id = labels.at<int>(i);
        Y_channel.at<float>(i) = centroids.at<float>(cluster_id);
    }

    // Reshape back to the original image shape
    Y_channel = Y_channel.reshape(1, src.rows);

    // Replace Y_channel in ycrcb_image with quantized_Y
    cv::Mat in2[] = { Y_channel };
    cv::Mat target2[] = { YCrCb };
    // mapping of input array channels to output array channels
    int from_to2[] = { 0, 0 };
    cv::mixChannels(in2, 1, target2, 1, from_to2, 1);

    //!NOTICE: must convert back before cvtColor, otherwise the BGR color will be wrong
    YCrCb.convertTo(YCrCb, CV_8U);


    // Convert back to BGR color space
    cv::cvtColor(YCrCb, dst, cv::COLOR_YCrCb2BGR);

    return(0);
}

/**
 * @brief 
 * extract edge from color image
 * edge is single channel binary image (black edge)  
 * @param src color image
 * @param dst single channel binary image
 * @return int 
 */
int edgeDetection(cv::Mat &src, cv::Mat &dst){

    cv::Mat gray,blur;
    blur.create(src.rows,src.cols,CV_8UC1);
    dst.create(src.rows,src.cols,CV_8UC1);

    // convert to grayscale 
    cvtColor(src, gray, COLOR_RGB2GRAY);
    // greyscale(src,gray);
    
    cv::medianBlur(gray,blur,7);
    
    
    // cv::medianBlur(gray,blur,5);
    // cv::Mat blur2;
    // blur2.create(src.rows,src.cols,CV_8UC1);
    // cv::bilateralFilter(blur,blur2,5,200,200);
    
    // imshow("m blur",blur);
    // imshow("b blur",blur2);

    int blocksize = 7;
    float c = 4;
    cv::adaptiveThreshold(blur,dst,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,5,2.5);
    // adaptiveThresholdMeanBinary(blur,dst,blocksize,c);

    return(0);
}

/**
 * @brief 
 * filter value into binary by threshold,
 * threshold is determined by small regions
 * @param src input should be single channel
 * @param dst single channel binary 
 * @return int 
 */
int adaptiveThresholdMeanBinary(cv::Mat &src, cv::Mat &dst, int blocksize, float constant){

    // check type is smae
    if(src.type()!=dst.type() || src.rows!=dst.rows){
        dst.create(src.rows,src.cols,CV_8UC1);}
    // check src is sinle channel

    int halfB = blocksize/2;
    
    for (int row = 0; row < src.rows; ++row){
        for (int col = 0; col < src.cols; ++col) {
            int sumOfBlock = 0;
            int count = 0;

            for (int brow = -halfB ; brow <= halfB; ++brow){
                for (int bcol = 0; bcol <= halfB; ++bcol){
                    int br = row+brow;
                    int bc = col+bcol;

                    if(br>=0 && br<src.rows && bc>=0 && bc<src.cols){
                        sumOfBlock += src.at<uchar>(br,bc);
                        count++;}
                }
            }
            // float mean = static_cast<float>(sumOfBlock)/count - constant;
            float mean = sumOfBlock/count - constant;
            dst.at<uchar>(row,col) = (src.at<uchar>(row,col) > mean)?255:0;
        }
    }
    return(0); 
}

/**
 * @brief 
 * cartoonize image by following steps
 * 1. blur image for blend details and sharpen edges
 * 2. luminance quantization
 * 3. edge detection
 * 4. combine quantization and edge
 * @param dst 
 * @param dst 
 * @return int 
 */
int cartoonize(cv::Mat &src, cv::Mat &dst){
    
    cv::Mat edge, lum_quant;
    cv::Mat edge3Channel;

    edge.create(src.rows,src.cols,CV_8UC1);
    lum_quant.create(src.rows,src.cols,CV_8UC3);

    // blur5x5_2(src,dst);
    bilateralFilter(src,dst,5,200,200);
    luminanceQuantization(dst,lum_quant,4);
    // imshow("luminance Quantized", lum_quant);

    edgeDetection(dst,edge);
    // imshow("edge", edge);

    // cv::Mat bilateral_quant;
    // bilateral_quant.create(src.rows,src.cols,CV_8UC3);
    // cv::bilateralFilter(lum_quant,bilateral_quant,7,200,200);

    cv::cvtColor(edge, edge3Channel, cv::COLOR_GRAY2BGR);
    // make sure the type() is same, inclue depth of bit and num of channel
    cv::bitwise_and(lum_quant, edge3Channel, dst);
    return(0);
}
/**
 * @brief 
 * apply cartoonization on detected face
 * @param src 
 * @param dst 
 * @param faces vec of Rects of detected faces
 * @param minWidth 
 * @return int 
 */
int cartoonizeface( cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces, int minWidth) {

    src.copyTo(dst);
    for(int i=0;i<faces.size();i++) {
    if( faces[i].width > minWidth ) {
        cv::Rect face( faces[i] );
        cv::Mat submatsrc = src(face);
        cv::Mat submatdst = dst(face);

        cartoonize(submatsrc,submatdst);
    }
  }

  return(0);
}