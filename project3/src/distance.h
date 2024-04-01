/*
  Author: Yilin Tang
  Date: 2024-02-24
  CS 5330 Computer Vision
  Description: 

*/

#ifndef distance_h
#define distance_h

// declarations here
enum class DistanceMetric {
    EuclideanDistance,
    CosineDistance
    // Add other distance metrics as needed
};

int distances_from_csv(std::vector<float> &targetFea, const char *csvfilename, std::vector<double> &distances, std::vector<char *> &labels, DistanceMetric metric = DistanceMetric::EuclideanDistance,float weight = 1.0);



#endif // distance_h
