#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "../include/shared.hpp"
#include "../include/marco_annunziata.hpp"
#include "../include/hermann_serain.hpp"

using namespace std;
using namespace cv;
using namespace HermannLib;
using namespace Shared;

// we can use everything in OpenCV except Deep Learning, we can use ML and can explore
// something different than what we've seen in the lectures (but there are no preferences)

vector<Mat> getModels(String pattern);
void testMetrics();

int main(int argc, char** argv){

    // if(argc < 3){
    //     cerr << "Usage: <test image path> <object_detection_dataset path>\n";
    //     return -1;

    // }
    // const string WINDOWNAME = "Test image";
    // string imgPath     = argv[1];
    // string datasetPath = argv[2];
    // const string drillModelsPath   = datasetPath + "035_" + Shared::toString(Shared::ImgObjType::power_drill)    + "/models/*.png";
    // const string sugarModelsPath   = datasetPath + "004_" + Shared::toString(Shared::ImgObjType::sugar_box)      + "/models/*.png";
    // const string mustardModelsPath = datasetPath + "006_" + Shared::toString(Shared::ImgObjType::mustard_bottle) + "/models/*.png";

    // string labelPath = imgPath;
    // labelPath = labelPath.replace(labelPath.find("test_images"), 11, "labels");
    // labelPath.replace(labelPath.find("-color.jpg"), 10, "-box.txt");

    // Mat testImg = imread(imgPath);
    // if(testImg.empty() == 1){
    //     cerr << "No image given as parameter\n";
    //     return -1;
    // }
    // namedWindow(WINDOWNAME);
    // imshow(WINDOWNAME, testImg);
    
    // vector<Mat> drillModels   = getModels(drillModelsPath);
    // vector<Mat> sugarModels   = getModels(sugarModelsPath);
    // vector<Mat> mustardModels = getModels(mustardModelsPath);

    // vector<string> labels;
    // string line;
    // ifstream labelFile;
    // labelFile.open(labelPath);
    // if(labelFile.is_open()){
    //     getline(labelFile, line);
    //     labels.push_back(line);
    // }
    // else{
    //     cerr << "No label file found";
    //     return -1;
    // }

    // pcaSift(&testImg, drillModels);

    //Test metrics
    testMetrics();

    waitKey(0);
    return(0);
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

    ObjMetric result = computeMetrics(testImgPath, ImgObjType::sugar_box, realRect, testRect);
    cout << result.toString() << endl;
}

vector<Mat> getModels(String pattern){
    const int MODELSCOUNT = 59;
    vector<Mat> models;
    vector<cv::String> modelsFilenames;
    glob(pattern, modelsFilenames, false);
    for(int i = 0; i < MODELSCOUNT; i++){ // TODO: maybe change this to read all images in the directory without a count limit
        models.push_back(imread(modelsFilenames[i]));
    }
    return models;
}
