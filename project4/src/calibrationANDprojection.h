/*
  Author: Yilin Tang
  Date: 2024-03-03
  CS 5330 Computer Vision
  Description: 

*/

#ifndef calibrationANDprojection_h
#define calibrationANDprojection_h


//declarations here
bool detectCorners(cv::Mat &src,std::vector<cv::Point2f>& corner_sets);
int calibrate(std::vector<std::vector<cv::Point3f>>& objectPoints, 
                       std::vector<std::vector<cv::Point2f>>& imagePoints, cv::Size imageSize, 
                       cv::Mat& cameraMatrix, cv::Mat& distCoeffs, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs);


bool calculate_camera_position_project_3DAxes(cv::Mat &frame, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, std::vector<cv::Point3f> &objectPoints);
bool projectVRobject(cv::Mat &frame, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                     const std::vector<cv::Point3f> &objectPoints, const cv::Size chessboardSize = cv::Size(9, 6));



#endif // calibrationANDprojection_h
