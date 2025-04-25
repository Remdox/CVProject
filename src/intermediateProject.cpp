#include <opencv2/core/types.hpp>
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
// something different from what we've seen the lectures

vector<modelView> getModelViews(string pattern);
void testMetrics();

int main(int argc, char** argv){

    if(argc < 3){
        cerr << "Usage: <test image path> <object_detection_dataset path>\n";
        return -1;

    }
    const string WINDOWNAME = "Test image";
    string imgPath     = argv[1];
    string datasetPath = argv[2];
    const string drillViewsPath   = datasetPath + "035_" + Shared::toString(Shared::ImgObjType::power_drill)    + "/models/*_color.png";
    const string sugarViewsPath   = datasetPath + "004_" + Shared::toString(Shared::ImgObjType::sugar_box)      + "/models/*_color.png";
    const string mustardViewsPath = datasetPath + "006_" + Shared::toString(Shared::ImgObjType::mustard_bottle) + "/models/*_color.png";

    string labelPath = imgPath;
    labelPath = labelPath.replace(labelPath.find("test_images"), 11, "labels");
    labelPath.replace(labelPath.find("-color.jpg"), 10, "-box.txt");

    Mat testImg = imread(imgPath);
    if(testImg.empty() == 1){
        cerr << "No image given as parameter\n";
        return -1;
    }
    namedWindow(WINDOWNAME);
    imshow(WINDOWNAME, testImg);

    objModel drillModel;
    objModel sugarModel;
    objModel mustardModel;
    drillModel.views   = getModelViews(drillViewsPath);
    sugarModel.views   = getModelViews(sugarViewsPath);
    mustardModel.views = getModelViews(mustardViewsPath);

    vector<string> labels;
    if(getLabels(&labels, labelPath) == -1){
        cerr << "No label file found";
        return -1;
    }

    vector<KeyPoint> testImgKpts = SIFT_PCA::detectKeypoints(testImg);
    Mat outputImg = testImg.clone();
    drawKeypoints(testImg, testImgKpts, outputImg);
    imwrite("../output/keypoints.png", outputImg);
    cout << "Loading drill models descriptors.." << endl;
    setViewsKeypoints(drillModel);
    cout << "Loading sugar models descriptors.." << endl;
    setViewsKeypoints(sugarModel);
    cout << "Loading mustard models descriptors.." << endl;
    setViewsKeypoints(mustardModel);
    outputImg = sugarModel.views.at(14).image.clone();
    drawKeypoints(sugarModel.views.at(14).image, sugarModel.views.at(14).keypoints, outputImg);
    imwrite("../output/modelKpts.png", outputImg);

    Mat descriptors;

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

vector<modelView> getModelViews(string pattern){
    vector<modelView> views;
    vector<string> viewsFilenames;
    glob(pattern, viewsFilenames, false);
    for(string modelFilename : viewsFilenames){
        modelView view;
        view.image = imread(modelFilename);
        if(view.image.empty()){
            cerr << "No image" << modelFilename << "found" << endl;
            continue;
        }
        views.push_back(view);
    }
    return views;
}
