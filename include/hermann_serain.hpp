#ifndef HERMANN_SERAIN_HPP
#define HERMANN_SERAIN_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <numeric>

using namespace cv;
using namespace std;

namespace HermannLib{

    class ObjMetric{
        private:
            ImgObjType type;
            double IoU;
            string sourceImg;
            bool isDetected;
    };

    double computeIoU(const Rect& a, const Rect& b);
    double computeMean(double vals[], int length);
}

#endif