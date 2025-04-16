#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

vector<Mat> getModels(String pattern){
    vector<Mat> models;
    vector<cv::String> modelsFilenames;
    glob(pattern, modelsFilenames, false);
    for(int i = 0; i < 59; i++){
        models.push_back(imread(modelsFilenames[i]));
    }
    return models;
}

// we can use everything in OpenCV except Deep Learning, we can use ML and can explore
// something different than what we've seen in the lectures (but there are no preferences)

int main(int argc, char** argv){
    if(argc < 3){
        cerr << "Usage: <test image path> <object_detection_dataset path>\n";
        return -1;
    
    }
    string imgPath     = argv[1];
    string datasetPath = argv[2];
    string labelPath = imgPath;
    labelPath = labelPath.replace(labelPath.find("test_images"), 11, "labels");
    labelPath.replace(labelPath.find("-color.jpg"), 10, "-box.txt");

    Mat testImg = imread(imgPath);
    if(testImg.empty() == 1){
        cerr << "No image given as parameter\n";
        return -1;
    }
    namedWindow("Test image");
    imshow("Test image", testImg);
    
    vector<Mat> drillModels   = getModels(datasetPath + "035_power_drill/models/*.png");
    vector<Mat> sugarModels   = getModels(datasetPath + "004_sugar_box/models/*.png");
    vector<Mat> mustardModels = getModels(datasetPath + "006_mustard_bottle/models/*.png");

    vector<string> labels;
    string line;
    ifstream labelFile;
    labelFile.open(labelPath);
    if(labelFile.is_open()){
        getline(labelFile, line);
        labels.push_back(line);
        cout << line << endl;
    }
    else{
        cerr << "No label file found";
        return -1;
    }

    waitKey(0);
    return(0);
}