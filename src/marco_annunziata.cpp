
#include "marco_annunziata.hpp"
#include "shared.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void getModelViews(string imgPattern, ObjModel& model){
    vector<string> viewsFilenames;
    glob(imgPattern, viewsFilenames, false);
    for(string modelFilename : viewsFilenames){
        modelView view;
        view.image = imread(modelFilename);
        string maskFilename = modelFilename.replace(modelFilename.find("color"), 5, "mask");
        view.mask = imread(maskFilename);
        if(view.image.empty()){
            cerr << "No image " << modelFilename << " found" << endl;
            continue;
        }
        model.views.push_back(view);
    }
}

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
        view.keypoints = SIFT_PCA::detectKeypoints(view.image, view.mask);
    }
}

void setViewsDescriptors(ObjModel& model){
    for(modelView& view : model.views){
        view.descriptors = SIFT_PCA::computeDescriptors(view.image, view.keypoints);
    }
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

    vector<KeyPoint> keypoints;
    sift->detect(grayImg, keypoints);
    return keypoints;
}

vector<KeyPoint> SIFT_PCA::detectKeypoints(Mat& img, Mat& mask) {
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    Mat grayMask;
    cvtColor(mask, grayMask, COLOR_BGR2GRAY);
    threshold(grayMask, grayMask, 1, 255, THRESH_BINARY); // Forza valori binari

    vector<KeyPoint> keypoints;
    sift->detect(grayImg, keypoints, grayMask);

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

