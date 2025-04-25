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

    // ObjMetric computeMetrics(...)
}