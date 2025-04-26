#include <opencv2/core/types.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
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

int main(int argc, char** argv){

    if(argc < 3){
        cerr << "Usage: <test image path> <object_detection_dataset path>\n";
        return -1;

    }
    const string WINDOWNAME = "Test image";
    string imgPath     = argv[1];
    string datasetPath = argv[2];
    const string drillViewsPath   = datasetPath + "035_" + toString(ImgObjType::power_drill)    + "/models/*_color.png";
    const string sugarViewsPath   = datasetPath + "004_" + toString(ImgObjType::sugar_box)      + "/models/*_color.png";
    const string mustardViewsPath = datasetPath + "006_" + toString(ImgObjType::mustard_bottle) + "/models/*_color.png";

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

    ObjModel drillModel;
    ObjModel sugarModel;
    ObjModel mustardModel;
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
    drawKeypoints(testImg, testImgKpts, outputImg, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite("../output/keypoints.png", outputImg);

    double t = (double)cv::getTickCount();
    cout << "Loading drill models descriptors.." << endl;
    setViewsKeypoints(drillModel);
    cout << "Loading sugar models descriptors.." << endl;
    setViewsKeypoints(sugarModel);
    cout << "Loading mustard models descriptors.." << endl;
    setViewsKeypoints(mustardModel);
    outputImg = sugarModel.views.at(14).image.clone();
    drawKeypoints(sugarModel.views.at(14).image, sugarModel.views.at(14).keypoints, outputImg, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite("../output/modelKpts.png", outputImg);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Overall time for feature detection: " << t << " seconds" << std::endl;

    t = (double)cv::getTickCount();
    setViewsDescriptors(drillModel);
    setViewsDescriptors(sugarModel);
    setViewsDescriptors(mustardModel);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Overall time for feature description: " << t << " seconds" << std::endl;

    ImgObjType objType = ImgObjType::sugar_box; //Leggiamo un altro param di input ?
    

    Mat descriptors;

    //Test metrics
    //Dopo aver fatto il matching, passare qua i 2 punti della bounding box
    //Rect rectFound = makeRect(/*xmin, ymin, xmax, ymax*/);
    Rect rectFound = makeRect(420, 300, 550, 450);
    ObjMetric metric = computeMetrics(imgPath, labelPath, objType, rectFound);
    metric.toString();

    waitKey(0);
    return(0);
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
