/*
  Author: Yilin Tang
  Date: 2024-02-24
  CS 5330 Computer Vision
  Description: 

  support train mode or test mode
  handle commands for video mode and image mode
  recognize and label object of single image
*/
#include <fstream>
#include <iterator> // For std::ostream_iterator
#include <filesystem>
#include <iostream>
#include <stdio.h>
#include <regex>
#include <opencv2/opencv.hpp>
#include "ORalgorithms.h"
#include "distance.h"
#include "csv_util.h"

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

static const char *train_mode = "train";
static const char *test_mode = "test";
static const char *image_mode = "image";
static const char *video_mode = "video";

/**
 * @brief 
 * extract label from filename
 * @param filename 
 * @return std::string 
 */
std::string extractLabel(std::string filename)
{
    std::regex pattern("^[a-zA-Z]+"); //  match one or more chars
    std::smatch matches;

    if (std::regex_search(filename, matches, pattern) && matches.size() > 0)
    {
        return matches[0];
    }
    return "unlabeled";
}

/**
 * @brief 
 * put label text to center of image
 * @param image 
 * @param text 
 */
void putTextCentered(cv::Mat &image, const std::string &text,cv::Scalar color)
{
    // Get the text size & baseline
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.8;
    int thickness = 2;
    int baseline = -50;
    cv::Size textboxSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // Calculate the starting point (bottom-left corner) to make the text centered
    cv::Point org((image.cols - textboxSize.width) / 2, (image.rows + textboxSize.height) / 2 - baseline);

    // Put the text in the image, Coral
    cv::putText(image, text, org, fontFace, fontScale,  color, thickness, cv::LINE_AA);
}

/**
 * @brief 
 * recognize object and check if the predicted is true
 * @param image 
 * @param DBfilenme 
 * @param actual_label 
 * @return std::string 
 */
std::string ORImg(cv::Mat &image, const char *DBfilenme, const char *actual_label)
{
    std::string label = classify(image, DBfilenme, DistanceMetric::CosineDistance, ClassificationMethod::KNN);

    std::ostringstream text;

    // label is unkown or video mode
    if(strcmp(actual_label, "unknown") == 0){
        putTextCentered(image, label.c_str(),cv::Scalar(255, 50, 100));
        return label;
    }

    text << "Actual: " << actual_label;
    text << ", Predicted: " << label.c_str();
    if (strcmp(actual_label, label.c_str()) == 0)
    {
        text << ", True";
        putTextCentered(image, text.str(),cv::Scalar(255, 50, 100));
    }
    else
    {
        text << ", False";
        putTextCentered(image, text.str(),cv::Scalar(214, 112, 218));
    }
    return label;
}

/**
 * @brief 
 * append feature of new image to csv Database
 * @param image 
 * @param DBfilenme 
 * @param label 
 * @return int 
 */
int trainORModel(cv::Mat &image, const char *DBfilenme, const char *label)
{

    std::vector<float> featureVec;

    // extract feature and label it to CSV
    int feanum = extractFeatureVector(image, featureVec, FeatureSize::variableSize);
    append_image_data_csv(DBfilenme, label, featureVec);

    printf("extracted %d feature\n", feanum);
    // print feature vector for test
    std::cout << std::fixed << std::setprecision(4);
    std::copy(featureVec.begin(), featureVec.end(), std::ostream_iterator<float>(std::cout, " | "));
    std::cout <<  std::endl;

    return 0;
}

/**
 * @brief 
 * wait a key for saving or other operation
 * @param labeled 
 * @param path 
 * @param filename 
 * @return int 
 */
int waitforSave(cv::Mat labeled, std::string path, std::string filename)
{  

    cv::Mat image = imread(path, 1);
    // stop to save data for image mode
    stringstream ss;
    char key = cv::waitKey(0);
    if (key == 's')
    {   
        ss.str("");
        ss << filename << "_labeled.jpg";
        imwrite(ss.str(), labeled);
        ss.str("");
        ss << filename << ".jpg";
        imwrite(ss.str(), image);
        ss.str("");
        cv::Mat thres, cleanup, segmentRegion, leastInertia;
        std::vector<std::vector<cv::Point>> tempSet;
        
        dynamicThreshold(image, thres);
        ss << filename << "_threshold.jpg";
        imwrite(ss.str(), thres);

        cleanupBinary(thres, cleanup);
        ss.str("");
        ss << filename << "_cleanup.jpg";
        imwrite(ss.str(), cleanup);

        int num = segmentIntoRegion(cleanup, segmentRegion, tempSet);  
        ss.str("");
        ss << filename << "_segment.jpg";
        imwrite(ss.str(), segmentRegion);
        
        vector<vector<float>> tempMap(num);
        leastInertiaInvariance(cleanup, leastInertia, tempSet, tempMap);
        ss.str("");
        ss << filename << "_leastInertia.jpg";
        imwrite(ss.str(), leastInertia);
        printf("save the original and preprocessed frame\n");
    }
    return 0;
}

/**
 * @brief 
 * handle commands for video or image mode 
 * @param args 
 * @param system_mode 
 * @return int 
 */
int handleORCommands(std::vector<std::string> &args, std::string system_mode)
{
    int argc = args.size();
    if (argc != 3 && argc != 4)
    {
        printf("wrong command length\n");
        return 1;
    }

    // path and file
    fs::path inputFile, dirPath, imagePath;
    const char *DBpath = args[argc - 1].c_str();
    std::vector<fs::path> file_paths;
    const char *task_mode = args[0].c_str();

    // image data
    inputFile = args[1];
    Mat image;
    namedWindow(task_mode, WINDOW_FREERATIO);

    // analysis data and save data
    std::map<std::string, std::map<std::string, int>> actual_predict_counts_map;
    int datasetSize = 0, TrueNum = 0;

    // check input is a file or dir
    if (!fs::exists(inputFile))
    {
        printf("input file Not exist\n");
    }
    else if (fs::is_directory(inputFile))
    {
        dirPath = inputFile;
    }
    else if (fs::is_regular_file(inputFile))
    {

        imagePath = inputFile;
        image = cv::imread(imagePath, 1);
    }

    // deal with namelabeled image dataset for train or test
    if (argc == 3 && !dirPath.empty())
    {
        dirPath = inputFile;

        // Collect all file paths
        for (const auto &entry : fs::directory_iterator(dirPath))
        {
            if (fs::is_regular_file(entry))
            { // Check if it's a file, not a directory
                file_paths.push_back(entry.path());
            }
        }

        // Sort the file paths alphabetically
        std::sort(file_paths.begin(), file_paths.end());

        // loop over all the images alphabetically
        for (const auto &path : file_paths)
        {
            printf("%s\n", path.filename().c_str());

            const char *actualLabel = extractLabel(path.filename().string()).c_str();
            image = imread(path.string(), 1);

            //// train or test mode
            if (strcmp(task_mode, train_mode) == 0)
            {
                trainORModel(image, DBpath, actualLabel);
            }
            if (strcmp(task_mode, test_mode) == 0)
            {
                std::string predicted = ORImg(image, DBpath, actualLabel);

                // count test result
                actual_predict_counts_map[actualLabel][predicted]++;
                datasetSize++;
                TrueNum += (strcmp(actualLabel, predicted.c_str()) == 0) ? 1 : 0;
            }
            // press to next one image or save
            imshow(task_mode, image);
            std::string name = path.filename();
            name.erase(name.size()-4); // remove .jpg
            waitforSave(image,path.string(),name);
        }
    }

    // deal with train image set of one class
    if (argc == 4 && !dirPath.empty() && strcmp(task_mode, train_mode) == 0)
    {

        // Collect all file paths
        for (const auto &entry : fs::directory_iterator(dirPath))
        {
            if (fs::is_regular_file(entry))
            { // Check if it's a file, not a directory
                file_paths.push_back(entry.path());
            }
        }
        // Sort the file paths alphabetically
        std::sort(file_paths.begin(), file_paths.end());

        // loop over all the images alphabetically
        for (const auto &path : file_paths)
        {
            std::string filename = path.filename().string();
            printf("%s\n", filename.c_str());

            image = imread(path.string(), 1);
            const char *actual_label = args[argc - 2].c_str();
            trainORModel(image, DBpath, actual_label);
            datasetSize++;    
            // press to next one or save
            imshow(task_mode, image);
            waitforSave(image,path.string(), filename);
        }
    }

    // deal with single image and DBfile for train or test
    if (dirPath.empty())
    {
        const char *actual_label;
        // train mode
        if (strcmp(task_mode, train_mode) == 0)
        {
            actual_label = extractLabel(imagePath.filename().string()).c_str();
            trainORModel(image, DBpath, actual_label);
        } // test mode
        if (strcmp(task_mode, test_mode) == 0)
        {
            actual_label = args[argc - 2].c_str();
            std::string predicted = ORImg(image, DBpath, actual_label);
        }
        imshow(task_mode, image);
    }

    // analyze test results, write confusion matrix
    if (strcmp(task_mode, test_mode) == 0 && !dirPath.empty())
    {
        std::ofstream outFile("confusion_matrix.md",std::ios::app);
        if (outFile.is_open())
        {
            writeConfusionMatrix(outFile, actual_predict_counts_map);
            outFile << "\nAccuracy: " << TrueNum << "/" << datasetSize;
            outFile.close();
        }
        else
        {
            std::cerr << "Failed to open file for writing." << std::endl;
        }
    }

    if(system_mode==image_mode){
        waitforSave(image,inputFile.string(), inputFile.filename().string());
    }

    return (0);
}
