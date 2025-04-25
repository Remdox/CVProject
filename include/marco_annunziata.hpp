#ifndef MARCO_ANNUNZIATA_HPP
#define MARCO_ANNUNZIATA_HPP

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "shared.hpp"
#include <iostream>
#include <vector>
#include <fstream>

struct modelView{
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct objModel{
    std::vector<modelView> views;
    Shared::ImgObjType type;
};

class SIFT_PCA{
    private:
        static cv::Ptr<cv::SIFT> sift;

    public:
        static std::vector<cv::KeyPoint> detectKeypoints(cv::Mat& img);
        static cv::Mat computeDescriptors(cv::Mat& img, std::vector<cv::KeyPoint> keypoints);
};

int getLabels(std::vector<std::string>* labels, std::string labelPath);
void getForegroundMask();
void setViewsKeypoints(objModel& model);
void setViewsDescriptors(objModel& model);

#endif
