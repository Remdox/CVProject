#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <numeric>

#include "./../include/hermann_serain.hpp"
#include "./../include/shared.hpp"

using namespace cv;
using namespace std;
using namespace Shared;

namespace HermannLib{

    const double threshold = 0.5;

    //Class methods
    //ObjMetric
    ObjMetric::ObjMetric(ImgObjType t, double iou, std::string sImg)
    : _type(t)
    , _IoU(iou)
    , _sourceImg(std::move(sImg)) 
    {
        
    }

    bool ObjMetric::isDetected() const{
        return _IoU > threshold;
    }

    string ObjMetric::toString() const{
        return "Object type: " + Shared::toString(_type) + " source img: " + _sourceImg + 
        " IoU: " + to_string(_IoU) + " is detected: " + to_string(isDetected());
    }

    //Static methods
    //To compute the bounding box I've used the cv::Rect class
    double computeIoU(const Rect& a, const Rect& b) {
        Rect inter = a & b;               // From the docs i can compute the intersection with just "&"
        double interArea = inter.area();
        double unionArea = a.area() + b.area() - interArea;
        if (unionArea <= 0) return 0.0;
        return interArea / unionArea;
    }
    
    //Given in input IoU values I'll have the mIoU
    double computeMean(double vals[], int length)
    {
        double result = 0;
        for(int i = 0; i < length; i++)
        result += vals[i];
        return result/length;
    }

    cv::Rect makeRect(int xmin, int ymin, int xmax, int ymax) {
        int width  = xmax - xmin;
        int height = ymax - ymin;
        // opzionale: controlli su width e height > 0
        return cv::Rect(xmin, ymin, width, height);
    }

    ObjMetric computeMetrics(string sourceImg, ImgObjType object, const Rect& a, const Rect& b)
    {
        double IoU = computeIoU(a,b);
        return ObjMetric(object, IoU, sourceImg);
    }
}