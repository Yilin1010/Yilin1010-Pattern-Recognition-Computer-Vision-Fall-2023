/*
  Author: Yilin Tang
  Date: 2024-02-10
  CS 5330 Computer Vision
  Description: 

*/

/**
 * @brief
 * Command:
 * target filename for T,
 * a directory of images as the database B,
 * the feature type,
 * the matching method,
 * and the number of images N to return
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <dirent.h>
#include <stdio.h>
#include <numeric>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include "csv_util.h"

namespace fs = std::filesystem;
using namespace cv;
/*
Given a directory on the command line, scans through the directory for image
files.
Prints out the full path name for each file. This can be used as an argument to
fopen or to cv::imread.
*/

double sevenSquare(const Mat &target, const Mat &compare);
double histIntersection(const std::vector<float> &feaVec1, const std::vector<float> &feaVec2);
int rgbHistfeature(const Mat &image, std::vector<float> &featureVector, int bins);
int centerHistfeature(const Mat &image, std::vector<float> &featureVector, int bins);
int topHistfeature(const Mat &image, std::vector<float> &featureVector, int bins);
int textureHistfeature(Mat &image, std::vector<float> &featureVector);
int mulhist_distances_from_csv(std::vector<std::vector<float>> &targetFeas, std::vector<char *> &csvfiles,
                               std::vector<float> &weights, std::map<double, std::string> &distancesmap);
int custom_DNN_distances_from_csv(char *targetname,std::vector<std::vector<float>> &targetFeas, std::vector<char *> &csvfiles,
                               std::vector<float> &weights, std::map<double, std::string> &distancesmap);

int main(int argc, char *argv[])
{
    fs::path dirPath = argv[1];
    std::vector<fs::path> file_paths;

    const char *wholecenterRGB = "wholecenterRGB";
    const char *topcenterRGB = "topcenterRGB";
    const char *texturecolor = "texturecolor";
    const char *DNNembeddings = "embeddings";
    const char *custom = "custom";
    

    // check for sufficient arguments
    if (argc < 2)
    {
        printf("usage: %s <directory path> <targetfilename> <featuretype>\n", argv[0]);
        printf("usage: %s <directory path> <targetfilename> <featuretype> -csv <csvfilepath>\n", argv[0]);
        printf("usage: %s <directory path> <targetfilename> <featuretype> -csv <csvfilepath> <csvfilepath> weight weight\n", argv[0]);

        exit(-1);
    }
    // get the directory path
    printf("Processing directory %s\n", dirPath.c_str());

    // get target image
    cv::Mat target, image;
    target = cv::imread((dirPath / argv[2]).c_str(), 1);
    cv::imshow("target", target);

    // get feature vector type, metrix type and weight
    std::string featuretype;
    featuretype = argv[3];
    std::vector<char *> csvfiles;
    std::vector<float> weights;
    // track if caculate target feature
    int ifGetTargetFeature = 0;
    if (argc >=5 && strcmp(argv[4], "-csv") == 0)
    {
        if (argc == 6)
        {
            csvfiles = {argv[5]};
        }
        if (argc >= 7)
        {
            csvfiles = {argv[5], argv[6]};
        }
        if (argc >= 9)
        {
            weights = {std::stof(argv[7]), std::stof(argv[8])};
        }
    }

    // rgb feature and distance data
    int bins = 8;
    std::map<double, std::string> distances;
    std::vector<float> feaVecT, feaVecC;
    double distance, sumofTarget;
    // wholecenterRGB feature data
    std::vector<float> wholefeaVecT, wholefeaVecC;
    std::vector<float> centerfeaT, centerfeaC;
    double wholedistance, centerdistance, sumwholeHistT, sumCenterHistT;
    // texture feature data
    std::vector<float> texturefeaVecT, texturefeaVecC;
    double texturedistance, sumtextureHistT;

    // csv files
    std::string csvfile;

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
        if (argc >= 6 && strcmp(argv[4], "-csv") == 0)
        {
            break;
        }
        std::string filename = path.filename().string();

        // check if the file is an image
        if (strstr(filename.c_str(), ".jpg") ||
            strstr(filename.c_str(), ".png") ||
            strstr(filename.c_str(), ".ppm") ||
            strstr(filename.c_str(), ".tif"))
        {

            // printf("processing image file: %s\n", dp->d_name);

            image = imread(path.string(), 1);
            // calculate the distance similarity of 7 by 7 center square for two image
            if (featuretype == "7")
            {
                distance = sevenSquare(target, image);
            }

            // calculate the histogram intersection for rgbwhole and rgbcenter
            if (featuretype == "rgbhist")
            {
                if (ifGetTargetFeature == 0)
                {
                    rgbHistfeature(target, feaVecT, bins);
                    sumofTarget = std::accumulate(feaVecT.begin(), feaVecT.end(), 0);
                    ifGetTargetFeature = 1;

                    csvfile = "rgbhist.csv";
                }
                rgbHistfeature(image, feaVecC, bins);
                distance = histIntersection(feaVecT, feaVecC);
                distance = distance / sumofTarget;

                // save feature vec
                append_image_data_csv(csvfile.c_str(), filename.c_str(), feaVecC, 0);

                // Clear to deallocate memory, otherwise give wrong distances
                // feaVecC.clear();
            }

            // calculate the weighted hist intersection for texture feature and whole RGB color feature
            if (featuretype == topcenterRGB)
            {
                topHistfeature(image, feaVecT, bins);
                centerHistfeature(image, centerfeaC, bins);

                // save feature vec
                append_image_data_csv("tophist.csv", filename.c_str(), feaVecT, 0);
                append_image_data_csv("centerhist.csv", filename.c_str(), centerfeaC, 0);
            }

            // calculate the weighted hist intersection for texture feature and whole RGB color feature
            if (featuretype == wholecenterRGB)
            {
                rgbHistfeature(image, wholefeaVecC, bins);
                centerHistfeature(image, centerfeaC, bins);

                // save feature vec
                append_image_data_csv("wholehist.csv", filename.c_str(), wholefeaVecC, 0);
                append_image_data_csv("centerhist.csv", filename.c_str(), centerfeaC, 0);
            }

            // calculate the half-half weighted hist intersection for rgbhist and center region hist
            if (featuretype == texturecolor)
            {
                rgbHistfeature(image, wholefeaVecC, bins);
                textureHistfeature(image, texturefeaVecC);

                // save feature vec
                append_image_data_csv("colorhist.csv", filename.c_str(), wholefeaVecC, 0);
                append_image_data_csv("texturehist.csv", filename.c_str(), texturefeaVecC, 0);
            }

            // save distance value
            distances.insert(std::make_pair(distance, filename.c_str()));
        }
    }
    printf("all distances computed\n");

    // ###########################
    // ######### compute feature vector from csv files
    if (argc >= 6 && strcmp(argv[4], "-csv") == 0)
    {
        if (featuretype == "rgbhist")
        {
            rgbHistfeature(target, feaVecT, bins);
            // calculate weighted feature
            std::vector<std::vector<float>> targetFeas = {feaVecT};
            csvfiles = {argv[5]};
            weights = {1.0};
            mulhist_distances_from_csv(targetFeas, csvfiles, weights, distances);

            /*
                        rgbHistfeature(target, feaVecT, bins);
                        sumofTarget = std::accumulate(feaVecT.begin(), feaVecT.end(), 0);

                        std::vector<char *> filenames;
                        std::vector<std::vector<float>> feactures;
                        read_image_data_csv(argv[5], filenames, feactures, 0);

                        for (std::size_t i = 0; i < filenames.size() && i < feactures.size(); ++i)
                        {
                            const auto &filename = filenames[i];
                            const auto &feacture = feactures[i];

                            distance = histIntersection(feaVecT, feacture);
                            distance = distance / sumofTarget;

                            // insert distance value to map
                            distances.insert(std::make_pair(distance, filename));
                        }
            */
            // check specific image
            // if(strstr(filename,"pic.1032")!= NULL ){
            //     printf("pic.1032.jpg %.2lf\n",distance);
            // }
        }

        if (featuretype == topcenterRGB)
        {
            // calculate target feature
            topHistfeature(target, feaVecT, bins);
            centerHistfeature(target, centerfeaT, bins);
            std::vector<std::vector<float>> targetFeas = {feaVecT, centerfeaT};

            if (weights.empty())
            {
                weights = {0.45, 0.55};
            }

            // calculate weighted feature
            mulhist_distances_from_csv(targetFeas, csvfiles, weights, distances);
        }

        if (featuretype == wholecenterRGB)
        {

            // calculate target feature
            rgbHistfeature(target, wholefeaVecT, bins);
            centerHistfeature(target, centerfeaT, bins);
            std::vector<std::vector<float>> targetFeas = {wholefeaVecT, centerfeaT};

            if (weights.empty())
            {
                weights = {0.45, 0.55};
            }
            // calculate weighted feature
            mulhist_distances_from_csv(targetFeas, csvfiles, weights, distances);
        }

        if (featuretype == texturecolor)
        {
            // calculate target feature
            rgbHistfeature(target, wholefeaVecT, bins);
            textureHistfeature(target, texturefeaVecT);

            std::vector<std::vector<float>> targetFeas = {wholefeaVecT, texturefeaVecT};
            if (weights.empty())
            {
                weights = {0.45, 0.55};
            }

            // calculate weighted feature
            mulhist_distances_from_csv(targetFeas, csvfiles, weights, distances);
        }
        //
        if (featuretype == DNNembeddings)
        {
            std::vector<std::vector<float>> targetFeas = {};
            weights = {1.0};
            custom_DNN_distances_from_csv(argv[2], targetFeas, csvfiles, weights, distances);
        }
        if (featuretype == custom)
        {   
            textureHistfeature(target, texturefeaVecT);
            std::vector<std::vector<float>> targetFeas = {texturefeaVecT};
            if (weights.empty())
            {
                weights = {0.6, 0.4};
            }
            custom_DNN_distances_from_csv(argv[2], targetFeas, csvfiles, weights, distances);
        }
        
    }

    // ###########################
    // ######### display top matches images
    int count = 0;
    int first = 4;
    int least = 5;
    cv::Mat smallestDisCmbined_image, largestDisCombined_image;

    // smallest distances from start
    for (auto pair = distances.begin(); pair != distances.end() && count <= first; ++pair, ++count)
    {
        printf("distance: %.2lf  %s\n", pair->first, pair->second.c_str());
        std::string fullpath = std::string(argv[1]) + '/' + pair->second;

        image = imread(fullpath, 1);
        if (smallestDisCmbined_image.empty())
        {
            smallestDisCmbined_image = image;
        }
        else
        {
            cv::hconcat(std::vector<cv::Mat>{smallestDisCmbined_image, image}, smallestDisCmbined_image);
        }
    }
    // largest distances from end
    count = 0;
    for (auto pair = distances.rbegin(); pair != distances.rend() && count <= least; ++pair, ++count)
    {
        printf("distance: %.6lf  %s\n", pair->first, pair->second.c_str());
        std::string fullpath = std::string(argv[1]) + '/' + pair->second;
        
        image = imread(fullpath, 1);
        if (largestDisCombined_image.empty())
        {
            largestDisCombined_image = image;
        }
        else
        {
            cv::hconcat(std::vector<cv::Mat>{largestDisCombined_image, image}, largestDisCombined_image);
        }
    }

   
    // show similar and dissimlar images
    // top matched is the smallest distance
    if (featuretype == "7" || featuretype == DNNembeddings)
    {
        imshow("top matched", smallestDisCmbined_image);
        imshow("end matched", largestDisCombined_image);
        imwrite((featuretype + "_Topmatches_" + argv[2]).c_str(), smallestDisCmbined_image);
        imwrite((featuretype + "_Endmatches_" + argv[2]).c_str(), largestDisCombined_image);

    }
    // top matched is the largest distance
    if (featuretype == "rgbhist" || featuretype == wholecenterRGB || featuretype == topcenterRGB || featuretype == texturecolor || featuretype == custom)
    {
        imshow("top matched", largestDisCombined_image);
        imshow("end matched", smallestDisCmbined_image);
        imwrite((featuretype + "_Topmatches_" + argv[2]).c_str(), largestDisCombined_image);
        imwrite((featuretype + "_Endmatches_" + argv[2]).c_str(), smallestDisCmbined_image);
    }

        cv::waitKey(0);
        return (0);
}