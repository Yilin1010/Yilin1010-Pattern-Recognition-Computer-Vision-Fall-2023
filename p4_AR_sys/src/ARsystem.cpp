/*
  Author: Yilin Tang
  Date: 2024-03-01
  CS 5330 Computer Vision
  Description:

*/

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include "calibrationANDprojection.h"

using namespace cv;
using namespace std;

int dispalyVid(int argc, char **argv)
{

    // INITIALIZE VIDEOCAPTURE
    VideoCapture *cap;

    // open video file
    if (argc == 1)
    {
        cap = new VideoCapture(0);
    }
    else if (argc == 2)
    {
        cap = new VideoCapture(argv[1]);
    }
    else
    {
        printf("wrong command for video stream\n");
    }

    // check if we succeeded
    if (!cap->isOpened())
    {
        cerr << "ERROR! Unable to open camera or video file\n";
        return -1;
    }

    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing"
         << "\nPress 'q' to terminate" << endl;

    // get some properties of the image
    Size refS((int)cap->get(cv::CAP_PROP_FRAME_WIDTH),
              (int)cap->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    namedWindow("original", 1); // identifies a window can be resized

    // user input
    char lastK = 0;
    stringstream ss;

    // display data
    int windowWidth = 500, windowHeight = 400;

    // image data
    cv::Mat frame, selected, modified;

    // 2d and 3d points data
    // camera parameters
    int actualSquare = 19; // mm for chessboard squares
    vector<Point2f> corners;
    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;
    Mat cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);
    // Set the diagonal elements
    cameraMatrix.at<double>(0, 0) = 1.0; // fx
    cameraMatrix.at<double>(1, 1) = 1.0; // fy
    cameraMatrix.at<double>(2, 2) = 1.0;
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
    vector<Mat> rvecs, tvecs;
    Size imageSize;
    vector<Point3f> objP;
    // Adjust these values based on your actual calibration pattern
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            objP.push_back(Point3f(j * actualSquare, i * actualSquare, 0));
        }
    }

    for (;;)
    {
        if (!cap->isOpened())
        {
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

        // // resize frame
        // cv::resize(frame, frame, Size(windowWidth, windowHeight));
        // // resize window
        // cv::resizeWindow("original", windowWidth, windowHeight);

        if (lastK == 0)
        {
            // cv::imshow("original", frame);
        }
        cv::imshow("original", frame);

        // waiting if there is a waiting keystroke
        char key = cv::waitKey(10);

        if (key != -1 && key != 'q' && key != 's')
        {
            lastK = key;
            frame.copyTo(selected);
            frame.copyTo(modified);
            imageSize = selected.size();
            // Set the principal point to be the center of the image
            cameraMatrix.at<double>(0, 2) = imageSize.width / 2.0;
            cameraMatrix.at<double>(1, 2) = imageSize.height / 2.0;
        }

        // exit
        if (key == 'q')
        {
            break;
        }

        // convert the current frame to greyscale
        if (key == 'g' || lastK == 'g')
        {
            cvtColor(selected, modified, COLOR_RGB2GRAY);
            cv::imshow("Video", modified);
        }

        // calibrate the camera
        if (key == 'c')
        {
            // detect coners
            bool success = detectCorners(modified, corners);
            if (success)
            { // Assuming 0 indicates success
                imagePoints.push_back(corners);
                objectPoints.push_back(objP);
            }
            else
            {
                cerr << "Corner detection failed for this selected image" << endl;
                // continue selection
                continue;
            }
            // update calibration
            calibrate(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

            cv::imshow("Detected Corners", modified);
            cv::waitKey(0);
        }

        // calculate the camera position(translateion and rotation) and project 3d to 2d
        // if (key == 'p' || lastK == 'p')
        if (key == 'p')
        {
            calculate_camera_position_project_3DAxes(modified, cameraMatrix, distCoeffs, objP);
        }
        // if (key == 't' || lastK == 't')
        if (key == 't')
        {
            projectVRobject(modified, cameraMatrix, distCoeffs, objP);
        }

        // save the current frame to file
        if (key == 's')
        {
            if (selected.empty())
            {
                imwrite("Clip.jpg", frame);
                printf("save the current frame to file\n");
            }
            else
            {
                ss << "org" << lastK << ".jpg";
                imwrite(ss.str(), selected);
                ss.clear();
                ss.str("");
                ss << lastK << ".jpg";
                imwrite(ss.str(), modified);
                ss.clear();
                ss.str("");
                printf("save the original and modified frame to file\n");
            }
        }
    }

    printf("video existed\n");
    cv::imshow("exist", frame);

    cv::waitKey(0);
    cv::destroyAllWindows();
    delete cap;
    return 0;
}

int main(int argc, char **argv)
{

    if (argc == 1 or argc == 2)
    {
        dispalyVid(argc, argv);
    }
    else
    {
        printf("Wrong command length\n");
        return -1;
    }
    return 0;
}
