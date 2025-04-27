#ifndef SVEVA_TUROLA_HPP
#define SVEVA_TUROLA_HPP

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "marco_annunziata.hpp"

std::vector<cv::Point> featureMatching(cv::Mat* img, ObjModel& models);

#endif