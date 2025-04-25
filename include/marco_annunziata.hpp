#ifndef MARCO_ANNUNZIATA_HPP
#define MARCO_ANNUNZIATA_HPP

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <fstream>

int getLabels(std::vector<std::string> *labels, std::string labelPath);
void getForegroundMask();
std::vector<cv::Mat> getModelsKeypoints(std::vector<cv::Mat> models);
cv::Mat pcaSift(cv::Mat* img);

#endif
