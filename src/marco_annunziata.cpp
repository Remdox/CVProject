
#include "marco_annunziata.hpp"
#include "shared.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

int getLabels(vector<string>* labels, string labelPath){
    string line;
    ifstream labelFile;
    labelFile.open(labelPath);
    if(!labelFile.is_open()) return -1;
    while(getline(labelFile, line)){
        labels->push_back(line);
    }
    labelFile.close();
    return 0;
}

void setViewsKeypoints(ObjModel& model){
    for(modelView& view : model.views){
        view.keypoints = SIFT_PCA::detectKeypoints(view.image);
    }
}

void setViewsDescriptors(ObjModel& model){
    for(modelView& view : model.views){
        view.descriptors = SIFT_PCA::computeDescriptors(view.image, view.keypoints);
    }
}

void getForegroundMask(){

}


// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ SIFT_PCA Class @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

// static variabile definition
Ptr<SIFT> SIFT_PCA::sift = SIFT::create(0, 3, 0.04, 10, 1.6, false);

// static methods
vector<KeyPoint> SIFT_PCA::detectKeypoints(Mat& img){
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    Mat mask = grayImg.clone(); // TODO: remove background from detection

    vector<KeyPoint> keypoints;
    sift->detect(grayImg, keypoints);
    return keypoints;
}


Mat SIFT_PCA::computeDescriptors(Mat& img, vector<KeyPoint> keypoints){
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    Mat descriptors;
    // TODO (WIP): sub with the real PCA-SIFT Algorithm described in the paper
    sift->compute(grayImg, keypoints, descriptors);
    return descriptors;
}

