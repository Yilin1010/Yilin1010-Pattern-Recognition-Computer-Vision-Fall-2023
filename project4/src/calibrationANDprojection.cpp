/*
  Author: Yilin Tang
  Date: 2024-03-03
  CS 5330 Computer Vision
  Description:

  calibrate and project 3d axes and VR objects to target
*/

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

/**
 * @brief 
 * 1. detectCorners
  input: a target chessboard or ARuconers of image
  output: vec<point> corner_sets/ vec<vec<point>>
  drawing corners on image
 * 
 * @param src 
 * @param corner_sets 
 * @return true 
 * @return false 
 */
bool detectCorners(cv::Mat &src, vector<cv::Point2f> &corner_sets)
{
  // convert image to binary
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);

  // specify number of corners
  int corners_per_row = 9;
  int corners_per_col = 6;
  bool found = cv::findChessboardCorners(gray, cv::Size(corners_per_row, corners_per_col),
                                         corner_sets, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

  // refine detection
  if (found)
  {
    cv::cornerSubPix(gray, corner_sets, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
    cv::drawChessboardCorners(src, cv::Size(corners_per_row, corners_per_col), corner_sets, found);
  }

  return found;
}

/**
 * @brief 
 * 
  2. select calibration
  user keypress 's'
  input: frame from camera
  output:
  vec<vec<point_2f> image_point_sets
  vec<vec<vec3f> world_point_sets
  each point set inside outer vector corresponding to one view(one image)
  save at least 5 frames as calibration images

  x - col, y - row
  careful about xyz direction in real world

  3. Calibrate the Camera
  input: intrinsic parameters(camera_matrix, distortion_ceofficients)
  output: re-projection error,rotations and translation

  reproject 2d points to real world 3d points
 * 
 * @param objectPoints 
 * @param imagePoints 
 * @param imageSize 
 * @param cameraMatrix 
 * @param distCoeffs 
 * @param rvecs 
 * @param tvecs 
 * @return int 
 */
int calibrate(vector<vector<Point3f>> &objectPoints,
              vector<vector<Point2f>> &imagePoints, Size imageSize,
              Mat &cameraMatrix, Mat &distCoeffs, vector<Mat> &rvecs, vector<Mat> &tvecs)
{

  // Check if we have enough images for calibration
  if (imagePoints.size() >= 5)
  {
    // Calibrate the camera
    double error = calibrateCamera(objectPoints, imagePoints, imageSize,
                                   cameraMatrix, distCoeffs, rvecs, tvecs,
                                   cv::CALIB_FIX_ASPECT_RATIO);

    // Print the camera matrix, distortion coefficients, and error
    cout << "\n\n## CALIBRATION ##" << endl;
    std::cout << std::fixed << std::setprecision(4);
    cout << imagePoints.size() << " images of view" << endl;
    cout << "Camera Matrix:\n"
         << cameraMatrix << endl;
    cout << "Distortion Coefficients:\n"
         << distCoeffs << endl;
    cout << "Re-projection Error: " << error << '\n'
         << endl;

    // Optionally, print rotations and translations
    // for (size_t i = 0; i < rvecs.size(); ++i)
    // {
    //   cout << "Rotation Vector [" << i << "]:\n"
    //        << rvecs[i] << endl;
    //   cout << "Translation Vector [" << i << "]:\n"
    //        << tvecs[i] << endl;
    // }
    return imagePoints.size();
  }
  else
  {
    cout << "Need at least 5 images for calibration. Currently have: " << imagePoints.size() << endl;
    return 1;
  }
}

/**
 * @brief 
4. calculate camera position
input: camera calibration parameters, selected image of view
output: real-time labeled image,rotation vector and traslate vector for each view

5. project to image
input: world point sets, rotation vector, translations vector

project corners into real world image in real time
or show initial 3d axes as camera move on

 * @param frame 
 * @param cameraMatrix 
 * @param distCoeffs 
 * @param objectPoints 
 * @return true 
 * @return false 
 */
bool calculate_camera_position_project_3DAxes(cv::Mat &frame, cv::Mat &cameraMatrix, cv::Mat &distCoeffs, std::vector<cv::Point3f> &objectPoints)
{
  // Convert frame to grayscale
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);

  // Try to detect corners in the frame
  std::vector<cv::Point2f> corners;
  bool found = cv::findChessboardCorners(gray, cv::Size(9, 6), corners);

  std::vector<cv::Point3f> axisPoints;
  // Origin
  axisPoints.push_back(cv::Point3f(0, 0, 0));
  // Points along the X, Y, Z axes
  axisPoints.push_back(cv::Point3f(100, 0, 0));  // X axis mm length
  axisPoints.push_back(cv::Point3f(0, 100, 0));  // Y axis mm length
  axisPoints.push_back(cv::Point3f(0, 0, -100)); // Z axis mm length (negative Z to project outwards)

  if (found)
  {
    // Refine corner locations
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

    // find camera position
    cv::Mat rvec, tvec;
    cv::solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

    // Convert rotation vector to rotation matrix
    cv::Mat Rmat;
    cv::Rodrigues(rvec, Rmat);

    std::cout << "\n\n## CAMERA POSE ##" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Rotation Matrix:\n"
              << Rmat << '\n';
    std::cout << "Translation Vector:\n"
              << tvec << std::endl;

    std::vector<cv::Point2f> projectedPoints;
    cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

    // Draw axes
    cv::line(frame, projectedPoints[0], projectedPoints[1], cv::Scalar(0, 0, 255), 3); // X axis in red
    cv::line(frame, projectedPoints[0], projectedPoints[2], cv::Scalar(0, 255, 0), 3); // Y axis in green
    cv::line(frame, projectedPoints[0], projectedPoints[3], cv::Scalar(255, 0, 0), 3); // Z axis in blue

    // Show the frame with the axes
    cv::imshow("Frame with 3D Axes", frame);
  }
  else
  {
    std::cout << "Target not found in the frame." << std::endl;
  }

  return found;
}

/**
 * @brief 
6. line the point in 3d point with 2d point
input: virtual line objects
output: image with virtual object

 * @param frame 
 * @param cameraMatrix 
 * @param distCoeffs 
 * @param objectPoints 
 * @param chessboardSize 
 * @return true 
 * @return false 
 */
bool projectVRobject(cv::Mat &frame, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
                     const std::vector<cv::Point3f> &objectPoints, const cv::Size chessboardSize)
{
  // Convert frame to grayscale
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  // Detect chessboard corners
  std::vector<cv::Point2f> corners;
  bool found = cv::findChessboardCorners(gray, chessboardSize, corners,
                                         cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

  if (!found)
  {
    std::cout << "Chessboard not found." << std::endl;
    return false;
  }

  // Refine corner detection
  cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
               TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.1));

  // find camera position
  cv::Mat rvec, tvec;
  cv::solvePnP(objectPoints, corners, cameraMatrix, distCoeffs, rvec, tvec);

  // calculate the center of the square and its size
  int squareIndex = 21;
  cv::Point3f squareCenter = (objectPoints[squareIndex] + objectPoints[squareIndex + 1] + 
                          objectPoints[squareIndex + chessboardSize.width] + 
                          objectPoints[squareIndex + chessboardSize.width + 1]) * 0.25;
  float squareSize = 19;

  // Adjust pyramid base size to fit within the square
  float halfBaseLength = squareSize / 8 * 3;         // The full length of one side of the base triangle
  float triangleHeight = sqrt(3.0) * halfBaseLength; // Height of an equilateral triangle
  float offsetFromCenter = triangleHeight / 3.0;     // To center the triangle
  float height = squareSize; 
  float abovePlaneHeight = 100;                  // Adjust the pyramid height as desired

  // Define the pyramid vertices above board relative to the square center
  std::vector<cv::Point3f> abovePyramidPoints;
  abovePyramidPoints.push_back(cv::Point3f(squareCenter.x, squareCenter.y + 2.0 * offsetFromCenter, -abovePlaneHeight));            // Base vertex 1
  abovePyramidPoints.push_back(cv::Point3f(squareCenter.x - halfBaseLength, squareCenter.y - offsetFromCenter, -abovePlaneHeight)); // Base vertex 2
  abovePyramidPoints.push_back(cv::Point3f(squareCenter.x + halfBaseLength, squareCenter.y - offsetFromCenter, -abovePlaneHeight)); // Base vertex 3
  abovePyramidPoints.push_back(cv::Point3f(squareCenter.x, squareCenter.y, -abovePlaneHeight-height));                           // Apex, assuming the height goes into the screen

  // Define the pyramid vertices relative to the square center
  std::vector<cv::Point3f> pyramidPoints;
  pyramidPoints.push_back(cv::Point3f(squareCenter.x, squareCenter.y + 2.0 * offsetFromCenter, -0));            // Base vertex 1
  pyramidPoints.push_back(cv::Point3f(squareCenter.x - halfBaseLength, squareCenter.y - offsetFromCenter, -0)); // Base vertex 2
  pyramidPoints.push_back(cv::Point3f(squareCenter.x + halfBaseLength, squareCenter.y - offsetFromCenter, -0)); // Base vertex 3
  pyramidPoints.push_back(cv::Point3f(squareCenter.x, squareCenter.y, -0-height));                           


  // Project pyramid points onto the image plane
  std::vector<cv::Point2f> imagePoints;
  std::vector<cv::Point2f> aboveImagePoints;
  cv::projectPoints(pyramidPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);
  cv::projectPoints(abovePyramidPoints, rvec, tvec, cameraMatrix, distCoeffs, aboveImagePoints);


  // Draw the above pyramid and pyramid on board plane
  for (int i=0;i<=2;++i){
    cv::line(frame, aboveImagePoints[i], aboveImagePoints[(i+1)%3], Scalar(0, 255, 0), 3);
    cv::line(frame, aboveImagePoints[i], aboveImagePoints[3], Scalar(0, 255, 0), 3);

    cv::line(frame, imagePoints[i], imagePoints[(i+1)%3], Scalar(0, 110, 0), 3);
    cv::line(frame, imagePoints[i], imagePoints[3], Scalar(0, 110, 0), 3);
  }

  // Draw lines between the two image space points
  for (int i =0;i<=3;++i){
    cv::line(frame, aboveImagePoints[i], imagePoints[i], Scalar(0, 255, 0), 1);
  }


  cv::imshow("triangular pyramid", frame);

  return true;
}


