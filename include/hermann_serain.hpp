#ifndef HERMANN_SERAIN_HPP
#define HERMANN_SERAIN_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <numeric>

#include "shared.hpp"

using namespace cv;
using namespace std;
using namespace Shared;

namespace HermannLib{

    class ObjMetric{
        private:
            ImgObjType _type;
            double _IoU;
            string _sourceImg;

        public:
            ObjMetric(ImgObjType t, double iou, string sImg);

        bool isDetected() const;
        string toString() const;

    };

    Rect makeRect(int xmin, int ymin, int xmax, int ymax);
    ObjMetric computeMetrics(string sourceImg, ImgObjType object, const Rect& a, const Rect& b);
}

#endif