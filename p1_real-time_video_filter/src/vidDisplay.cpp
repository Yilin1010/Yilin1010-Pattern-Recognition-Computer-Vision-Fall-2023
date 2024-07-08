/*
  Author: Yilin Tang
  Date: 2024-01-25
  CS 5330 Computer Vision
  Description: 

  grab and display frame from video stream
  program take different keypress to 
    1. apply multiple filters to selected image from video
    2. save modified image
  usage: vidDisplay.out or vidDisplay.out <video_Path>
  compile: -o executable vidDisplay.cpp filter.cpp
*/

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "faceDetect.h"
#include "filter.h"

using namespace cv;
using namespace std;


/**
 * @brief 
 * modification key [g h p b x y m l t n e c a] 
 * quit/save key [q s]
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char **argv)
{

    // INITIALIZE VIDEOCAPTURE
    VideoCapture *cap;
    // open video file
    if (argc == 1)
    {
        cap = new VideoCapture(0);
    }else if (argc == 2)
    {
        cap = new VideoCapture(argv[1]);
    }{
        printf("usage: vidDisplay.out <video_Path>\n");
    }    

    // check if we succeeded
    if (!cap->isOpened())
    {
        cerr << "ERROR! Unable to open camera or video file\n";
        return -1;
    }

    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
         << "Press 'q' to terminate" << endl;

    // get some properties of the image
    Size refS((int)cap->get(cv::CAP_PROP_FRAME_WIDTH),
              (int)cap->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    namedWindow("Video", 1); // identifies a window



    Mat frame,selected;
    Mat modified;
    char lastK = 0;
    stringstream ss;

    // for face 
    cv::Mat grey;
    std::vector<cv::Rect> faces;
    cv::Rect last(0, 0, 0, 0);

    for (;;)
    {   
        if(!cap->isOpened()){
            cap->set(cv::CAP_PROP_POS_FRAMES, 0); // Loop back to start
        }
        *cap >> frame; // get a new frame from the VideoCapture, treat as a stream

        if (frame.empty())
        {
            // printf("frame is empty\n");
            // break;
            printf("Loop back\n");
            cap->set(cv::CAP_PROP_POS_FRAMES, 0); // Loop back to start
            continue;
        }
        imshow("latest frame", frame);

        // see if there is a waiting keystroke
        char key = waitKey(10);
        if(key!=-1 && key!='q'&& key!='s'){
            lastK = key;
            frame.copyTo(selected);
        }

        // exit
        if (key == 'q'){break;}

        // convert the current frame to greyscale
        if (key == 'g')
        {
            cvtColor(frame, modified, COLOR_RGB2GRAY);
            imshow("grayscale", modified);
        }

        // convert currenct frame to another grayscale
        if (key == 'h')
        {
            greyscale(frame,modified);
            imshow("another grayscale",modified);
        }

        // convert currenct frame to septia tone version
        if (key == 'p')
        {
            sepiatone1(frame, modified);
            imshow("septia tone frame", modified);
        }
        
        if (key == 'b')
        {
            blur5x5_1(frame, modified);
            imshow("blur", modified);
        }
        

        // 'x' key shows the X Sobel, 
        // 'y' key shows the Y Sobel.
        if (key == 'x')
        {
            sobelX3x3(frame, modified);
            convertScaleAbs(modified,modified);
            imshow("X Sobel", modified);
        }
        if (key == 'y')
        {
            sobelY3x3(frame, modified);
            convertScaleAbs(modified,modified);
            imshow("Y Sobel", modified);
        }

        // 'm' key shows the color gradient magnitude image.
        if (key == 'm')
        {
            Mat x, y;
            sobelX3x3(frame, x);
            sobelY3x3(frame, y);
            convertScaleAbs(x,x);
            convertScaleAbs(y,y);
            magnitude(x,y,modified);
            imshow("gradient magnitude", modified);
        }


        // 'the 'l' key displays blurQuantize image.
        if (key == 'l')
        {
            blurQuantize(frame, modified, 10);
            imshow("blur Quantize", modified);
        }

        // 'the 'f' key detect faces in an image
        if (key == 'f')
        {
            // convert the image to greyscale
            cv::cvtColor( frame, grey, cv::COLOR_BGR2GRAY, 0);

            // detect faces
            detectFaces( grey, faces );

            // draw boxes around the faces
            drawBoxes( frame, faces );

            // add a little smoothing by averaging the last two detections
            if( faces.size() > 0 ) {
                last.x = (faces[0].x + last.x)/2;
                last.y = (faces[0].y + last.y)/2;
                last.width = (faces[0].width + last.width)/2;
                last.height = (faces[0].height + last.height)/2;
            }

            // display the frame with the box in it
            cv::imshow("Video", frame);
            // record detected frame
            frame.copyTo(modified);
        }

        // 'the 't' key displays binary adaptive threshold
        if (key == 't')
        {
            cv::Mat gray;
            cvtColor(frame, gray, COLOR_RGB2GRAY);
            adaptiveThresholdMeanBinary(gray, modified,3,1);
            imshow("adaptive threshold", modified);
        }

        // 'the 'n' key displays luminance Quantization image.
        if (key == 'n')
        {
            luminanceQuantization(frame, modified,4);
            imshow("luminance Quantization", modified);
        }

        // 'the 'e' key displays edge detector image.
        if (key == 'e')
        {
            edgeDetection(frame, modified);
            imshow("edge detector", modified);
        }

        // 'the 'c' key to cartoonize whole image.
        if (key == 'c')
        {            
            cartoonize(frame, modified);
            imshow("cartoonized", modified);
        }


        // 'the 'a' key to cartoonize face.
        if (key == 'a')
        {            
            // convert the image to greyscale
            cv::cvtColor( frame, grey, cv::COLOR_BGR2GRAY, 0);

            // detect faces
            detectFaces( grey, faces );
            cartoonizeface(frame, modified, faces);
            imshow("cartoonized", modified);
        }



        // printf("press 'q' to exit, 's' to save, any other keys to continue\n");

        // save the current frame to file
        if (key == 's')
        {   
            if(modified.empty()){
                imwrite("Clip.jpg",frame);
                printf("save the current frame to file\n");
            }else{
                ss << "org" << lastK << ".jpg";
                imwrite(ss.str(),selected);
                ss.clear();
                ss.str("");
                ss <<  lastK << ".jpg";
                imwrite(ss.str(),modified);
                ss.clear();
                ss.str("");
                printf("save the original and modified frame to file\n");
            }

        }
    }

    printf("show last frame\n");
    imshow("last frame", frame);

    waitKey(0);
    delete cap;
    return (0);
}
