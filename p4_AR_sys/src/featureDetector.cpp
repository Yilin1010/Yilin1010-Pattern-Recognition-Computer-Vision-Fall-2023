/*
  Author: Yilin Tang
  Date: 2024-03-19
  CS 5330 Computer Vision
  Description: 
  
  apply Harris corners and ORB and draw keypoints to video stream
*/

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

// Function to perform feature detection
void featureDetector(cv::Mat image)
{
  // Convert the image to grayscale
  cv::Mat grayImage;
  cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

  // Initialize ORB detector
  // The maximum number of features to retain.
  // scaleFactor. the original image will be downscaled
  // The number of pyramid levels, depends on range of feature size, pyramid is a stack of scaled image
  // edgeThreshold, the width of edges, so feature within edge won't be detected
  // Low fastThreshold, subtle variations in intensity detected, 
  int nlevels = 4;
  int edgeThreshold = 30;
  int patchsize = 30;
  cv::Ptr<cv::ORB> orbDetector = cv::ORB::create(50,1.1,nlevels,edgeThreshold,0,2,
  cv::ORB::HARRIS_SCORE,patchsize,20);

  // Detect ORB features
  std::vector<cv::KeyPoint> keypointsORB;
  cv::Mat descriptorsORB;
  orbDetector->detectAndCompute(grayImage, cv::Mat(), keypointsORB, descriptorsORB);

  // Draw ORB keypoints
  cv::Mat imageORB;
  cv::drawKeypoints(image, keypointsORB, imageORB, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  // Harris corner detection
  cv::Mat harrisCorners, harrisNormalized;
  // 
  cv::cornerHarris(grayImage, harrisCorners,5,9, 0.2, cv::BORDER_DEFAULT);
  cv::normalize(harrisCorners, harrisNormalized, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

  // Draw Harris corners
  cv::Mat imageHarris = image.clone();
  for (int i = 0; i < harrisNormalized.rows; i++)
  {
    for (int j = 0; j < harrisNormalized.cols; j++)
    {
      if ((int)harrisNormalized.at<float>(i, j) > 125)
      {
        cv::circle(imageHarris, cv::Point(j, i),5, cv::Scalar(0, 0, 255),2);
      }
    }
  }

  // Display results
  cv::imshow("ORB Features", imageORB);
  cv::imshow("Harris Corners", imageHarris);
  // cv::waitKey(0);
}

int main(int argc, char **argv)
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

  namedWindow("video", 1); // identifies a window can be resized

  // user input
  char lastK = 0;
  stringstream ss;

  // display data
  int windowWidth = 810, windowHeight = 540;
  // resize window
  cv::resizeWindow("video", windowWidth, windowHeight);

  // image data
  cv::Mat frame, selected, modified;

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

    // resize frame
    cv::resize(frame, frame, Size(windowWidth, windowHeight));

    if (lastK == 0)
    {
      // cv::imshow("video", frame);
    }
    cv::imshow("video", frame);

    // waiting if there is a waiting keystroke
    char key = cv::waitKey(10);

    if (key != -1 && key != 'q' && key != 's')
    {
      lastK = key;
      frame.copyTo(selected);
      frame.copyTo(modified);
    }

    // exit
    if (key == 'q')
    {
      break;
    }

    if (key == 'f' || lastK == 'f')
    {
      // Perform feature detection
      featureDetector(frame);
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
        imwrite(ss.str(), selected);
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
