/*
  Author: Yilin Tang
  Date: 2024-01-25
  CS 5330 Computer Vision
  Description: 

  time different image processing tasks using different computation methods 
  Program takes a path to an image on the command line
*/

#include <cstdio> // a bunch of standard C/C++ functions like printf, scanf
#include <cstring> // C/C++ functions for working with strings
#include <cmath>
#include <sys/time.h>
#include "opencv2/opencv.hpp"
#include <functional>

// prototypes for the functions to test
int blur5x5_1( cv::Mat &src, cv::Mat &dst );
int blur5x5_2( cv::Mat &src, cv::Mat &dst );
int sepiatone1(cv::Mat &src, cv::Mat &dst);
int sepiatone2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );

// returns a double which gives time in seconds
double getTime() {
  struct timeval cur;

  gettimeofday( &cur, NULL );
  return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}

/**
 * @brief 
 * a helper to evaluate and print time given different func refer
 * @param evalFunc callable function object
 * @param src 
 * @param dst 
 */
void evalTime(std::function<int(cv::Mat&, cv::Mat&)> evalFunc,cv::Mat &src, cv::Mat &dst){
    const int Ntimes = 100;
    printf("execute %d times\n", Ntimes);

    // set up the timing for version 1
    double startTime = getTime();

    // execute the file on the original image a couple of times
    for(int i=0;i<Ntimes;i++) {
    evalFunc(src, dst);
    }

    // end the timing
    double endTime = getTime();

    // compute the time per image
    double difference = (endTime - startTime) / Ntimes;

    // print the results
    printf("Time per image (1): %.4lf seconds\n", difference );

}
  

// argc is # of command line parameters (including program name), argv is the array of strings
// This executable is expecting the name of an image on the command line.

int main(int argc, char *argv[]) {  // main function, execution starts here
    cv::Mat src; // define a Mat data type (matrix/image), allocates a header, image data is null
    cv::Mat dst; // cv::Mat to hold the output of the process
    char filename[256]; // a string for the filename

    // usage: checking if the user provided a filename
    if(argc < 2) {
    printf("Usage %s <image filename>\n", argv[0]);
    exit(-1);
    }
    strcpy(filename, argv[1]); // copying 2nd command line argument to filename variable

    // read the image
    src = cv::imread(filename); // allocating the image data
    // test if the read was successful
    if(src.data == NULL) {  // src.data is the reference to the image data
    printf("Unable to read image %s\n", filename);
    exit(-1);}

    // // evaluate blur
    // printf("Time for pixel-wise blur\n");
    // evalTime(blur5x5_1,src, dst);
    // evalTime(blur5x5_2,src, dst);

    // evaluate sepiatone
    printf("Time for pixel-wise sepiatone\n");
    evalTime(sepiatone1,src, dst);
    evalTime(sepiatone2,src, dst);

  return(0);
}



