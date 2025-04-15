#include <stdio.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

vector<Mat> getModels(String pattern){
    vector<Mat> models;
    vector<cv::String> modelsFilenames;
    glob(pattern, modelsFilenames, false);
    for(int i = 0; i < 60; i++){
        models.push_back(imread(modelsFilenames[i]));
    }
    return models;
}

// we can use everything in OpenCV except Deep Learning, we can use ML and can explore
// something different than what we've seen in the lectures (but there are no advantages)

int main(int argc, char** argv){
    Mat imgDest;

    if(argc < 4){
        cerr << "Usage: path_image path_models_directory path_labels_directory\n";
        return -1;
    }
    
    Mat testImg = imread(argv[1]);
    if(testImg.empty() == 1){
        cerr << "No image given as parameter\n";
        return -1;
    }
    namedWindow("Test image");
    imshow("Test image", testImg);
    
    vector<Mat> modelsDrill = getModels("/object_detection_dataset/035_power_drill/models/*.png");
    vector<Mat> modelsSugar = getModels("/object_detection_dataset/004_sugar_box/models/*.png");
    vector<Mat> modelsMustard = getModels("/object_detection_dataset/006_mustard_bottle/models/*.png");
    


    

    waitKey(0);
    return(0);
}