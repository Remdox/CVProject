#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "../include/marco_annunziata.hpp"

using namespace std;
using namespace cv;

// we can use everything in OpenCV except Deep Learning, we can use ML and can explore
// something different than what we've seen in the lectures (but there are no preferences)

vector<Mat> getModels(String pattern);

int main(int argc, char** argv){
    if(argc < 3){
        cerr << "Usage: <test image path> <object_detection_dataset path>\n";
        return -1;

    }
    const string WINDOWNAME = "Test image";
    string imgPath     = argv[1];
    string datasetPath = argv[2];
    const string drillModelsPath = datasetPath + "035_power_drill/models/*.png";
    const string sugarModelsPath = datasetPath + "004_sugar_box/models/*.png";
    const string mustardModelsPath = datasetPath + "006_mustard_bottle/models/*.png";
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
    
    vector<Mat> drillModels   = getModels(drillModelsPath);
    vector<Mat> sugarModels   = getModels(sugarModelsPath);
    vector<Mat> mustardModels = getModels(mustardModelsPath);

    vector<string> labels;
    string line;
    ifstream labelFile;
    labelFile.open(labelPath);
    if(labelFile.is_open()){
        getline(labelFile, line);
        labels.push_back(line);
    }
    else{
        cerr << "No label file found";
        return -1;
    }

    waitKey(0);
    return(0);
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
