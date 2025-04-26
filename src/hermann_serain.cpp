#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <numeric>
#include <fstream>
#include <sstream>
#include <array>
#include <stdexcept>

#include "./../include/hermann_serain.hpp"
#include "./../include/shared.hpp"

using namespace cv;
using namespace std;
using namespace Shared;

namespace HermannLib{

    const double threshold = 0.5;

    //prototypes
    std::vector<std::array<int,4>> extractBoundingBoxes(const std::string& filePath, const std::string& objName);

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

    ObjMetric computeMetrics(string sourceImgPath, string labelPath, ImgObjType object, const Rect& a)
    {
        vector<array<int,4>>  values = extractBoundingBoxes(labelPath, toString(object));
        if(values.size() == 0)
            throw std::runtime_error("Non ci sono oggetti di quel tipo in questa immagine");
        Rect b = makeRect(values[0][0],values[0][1],values[0][2],values[0][3]);
        double IoU = computeIoU(a,b);
        return ObjMetric(object, IoU, sourceImgPath);
    }

    vector<std::array<int,4>> extractBoundingBoxes(const std::string& filePath, const std::string& objName)
    {
        std::ifstream infile(filePath);
        if (!infile.is_open()) {
            throw std::runtime_error("Impossibile aprire il file: " + filePath);
        }
    
        std::vector<std::array<int,4>> result;
        std::string line;
        size_t lineNum = 0;
        while (std::getline(infile, line)) {
            ++lineNum;
            auto pos = line.find(' '); // separa al primo spazio
            std::string key = (pos == std::string::npos ? line : line.substr(0, pos));
            std::string values = (pos == std::string::npos ? "" : line.substr(pos+1));
    
            if (key.find(objName) != std::string::npos) {
                std::istringstream iss(values);
                std::array<int,4> coords;

                for (int i = 0; i < 4; ++i) {
                    if (!(iss >> coords[i])) {
                        throw std::runtime_error(
                            "Parsing error a linea " + std::to_string(lineNum)
                            + ": attesi 4 interi, trovati solo " 
                            + std::to_string(i));
                    }
                }
            }
        }
    
        return result;
    }

    void testMetrics(){
        string testImgPath = "./../data/object_detection_dataset/004_sugar_box/test_images/4_0001_000121-color.jpg";
        int xmin = 397, ymin = 235, xmax = 469, ymax = 460; //4_0001_000121-box.txt
        //int xmintest = 397, ymintest = 235, xmaxtest = 469, ymaxtest = 460; //100%
        //int xmintest = 397, ymintest = 235, xmaxtest = 469, ymaxtest = 400; //>50%
        // int xmintest = 397, ymintest = 235, xmaxtest = 469, ymaxtest = 347; //50%
        //int xmintest = 397, ymintest = 235, xmaxtest = 469, ymaxtest = 300; //<50%
        //int xmintest = 480, ymintest = 300, xmaxtest = 550, ymaxtest = 450; //0%
        //Random points di chatgpt
        int xmintest = 420, ymintest = 300, xmaxtest = 550, ymaxtest = 500;  
        Rect realRect = makeRect(xmin, ymin, xmax, ymax);
        Rect testRect = makeRect(xmintest, ymintest, xmaxtest, ymaxtest);
    
        // ObjMetric result = computeMetrics(testImgPath, ImgObjType::sugar_box, realRect, testRect);
        // cout << result.toString() << endl;
    }
}