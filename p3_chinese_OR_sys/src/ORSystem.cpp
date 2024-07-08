/*
  Author: Yilin Tang
  Date: 2024-02-10
  CS 5330 Computer Vision
  Description:

  support video mode or image mode,
  dispaly and recognize objects in real-time video
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
#include "ORalgorithms.h"

using namespace cv;
using namespace std;

std::string ORImg(cv::Mat &image, const char *DBfilenme, const char *actual_label);
int trainORModel(cv::Mat &image, const char *DBfilenme, const char *label);
int handleORCommands(std::vector<std::string> &args, std::string system_mode);

int dispalyVid(int argc, char **argv)
{

    // INITIALIZE VIDEOCAPTURE
    VideoCapture *cap;

    // open video file
    if (argc == 2)
    {
        cap = new VideoCapture(0);
    }
    else if (argc == 3)
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
    cout << "Start grabbing" << endl
         << "Press 'q' to terminate" << endl;

    // get some properties of the image
    Size refS((int)cap->get(cv::CAP_PROP_FRAME_WIDTH),
              (int)cap->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d %d\n", refS.width, refS.height);

    namedWindow("original", 1); // identifies a window can be resized

    // data for features
    Mat oriframe,frame, selected, modified;
    Mat segmentRegion, leastInertia;
    char lastK = 0;
    stringstream ss;
    std::vector<std::vector<cv::Point>> rectBoxs;

    // video and DB data
    std::string system_mode = "train";
    const char *DBfile = argv[argc - 1];
    int windowWidth = 500, windowHeight = 400;

    for (;;)
    {
        if (!cap->isOpened())
        {
            cap->set(cv::CAP_PROP_POS_FRAMES, 0); // Loop back to start
        }
        *cap >> oriframe; // get a new frame from the VideoCapture, treat as a stream
        if (oriframe.empty())
        {
            // printf("frame is empty\n");
            // break;
            printf("Loop back\n");
            cap->set(cv::CAP_PROP_POS_FRAMES, 0); // Loop back to start
            continue;
        }

        // resize frame
        cv::resize(oriframe, frame, Size(windowWidth, windowHeight));
        // resize window
        cv::resizeWindow("original", windowWidth, windowHeight);

        if (lastK == 0)
        {
            cv::imshow("original", frame);
        }

        // waiting if there is a waiting keystroke
        char key = cv::waitKey(10);

        if (key != -1 && key != 'q' && key != 's')
        {
            lastK = key;
            frame.copyTo(selected);
        }

        // exit
        if (key == 'q')
        {
            break;
        }

        // convert the current frame to greyscale
        if (key == 'g' || lastK == 'g')
        {
            cvtColor(frame, modified, COLOR_RGB2GRAY);
            cv::imshow("Video", modified);
        }

        // OR the frame
        if (key == 'r' || lastK == 'r')
        {
            ORImg(frame, DBfile, "unknown");
            cv::imshow("video", frame);
        }

        // save the current frame to file
        if (key == 's')
        {
            if (modified.empty())
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

    printf("show last frame\n");
    cv::imshow("last frame", frame);

    cv::waitKey(0);
    delete cap;
    return (0);
}

/**
 * @brief
 * quit/save key [q s]
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char **argv)
{

    if (argc == 2 or argc == 3)
    {
        dispalyVid(argc, argv);
    }
    else if (argc != 4 && argc != 5)
    {
        printf("usage: ORSystem.out <DB_Path>\n");
        printf("usage: ORSystem.out <video_Path> <DB_Path>\n");
        printf("usage: ORSystem.out train <Dir_Path> <DB_Path>\n");
        printf("usage: ORSystem.out train <Dir_Path> <actual_label> <DB_Path>\n");
        printf("usage: ORSystem.out train <Image_Path> <actual_label> <DB_Path>\n");
        printf("usage: ORSystem.out test <Image_Path> <DB_Path>\n");
        printf("usage: ORSystem.out test <Dir_Path> <DB_Path>\n");
        return -1;
    }
    {
        std::vector<std::string> args(argv + 1, argv + argc);
        handleORCommands(args, "image");
    }
}
