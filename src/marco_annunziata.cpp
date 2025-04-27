
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
Ptr<SIFT> SIFT_PCA::sift = SIFT::create(0,
                                        3,
                                        0.04,
                                        10,
                                        1.6,
                                        false);

// static methods
vector<KeyPoint> SIFT_PCA::detectKeypoints_canny(Mat& img) {
    Mat grayImg, edges, mask;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    Canny(grayImg, edges, 50, 150);
    dilate(edges, mask, Mat(), Point(-1,-1), 3);

    vector<KeyPoint> keypoints;
    sift->detect(grayImg, keypoints, mask);
    return keypoints;
}

vector<KeyPoint> SIFT_PCA::detectKeypoints(Mat& img){
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    Mat mask = grayImg.clone(); // TODO: remove background from detection

    vector<KeyPoint> keypoints;
    sift->detect(grayImg, keypoints);
    return keypoints;
}

vector<KeyPoint> SIFT_PCA::detectKeypoints_grabCut(Mat& img){
    Mat grayImg;
    Mat mask(img.size(), CV_8UC1, GC_BGD); // TODO: remove background from detection
    Mat bgdModel, fgdModel;
    Rect RegionOfInterest(img.cols * 0.1, img.rows * 0.1, img.cols * 0.8, img.rows * 0.8);
    rectangle(mask, RegionOfInterest, Scalar(GC_PR_FGD), -1);

    grabCut(img, mask, RegionOfInterest, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);

    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    Mat foreground;
    grayImg.copyTo(foreground, (mask == GC_FGD) | (mask == GC_PR_FGD));

    vector<KeyPoint> keypoints;
    sift->detect(foreground, keypoints);
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

