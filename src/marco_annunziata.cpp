
#include "marco_annunziata.hpp"
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

int getLabels(vector<string> *labels, string labelPath){
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

void getForegroundMask(){

}

vector<Mat> getModelsKeypoints(vector<Mat> models){
    vector<Mat> modelKeypoints;
    for(Mat modelView : models){
        modelKeypoints.push_back(pcaSift(&modelView));
    }
    return modelKeypoints;
}

Mat pcaSift(Mat* img){
    Mat grayImg = img->clone();
    cvtColor(*img, grayImg, COLOR_BGR2GRAY);
    Ptr<SIFT> sift = SIFT::create(0, 3, 0.04, 10, 1.6, false);
    vector<KeyPoint> keypoints;
    sift.get()->detect(grayImg, keypoints, noArray());

    sift.get()->compute(grayImg, keypoints, noArray()); // TODO (WIP): sub with the real PCA-SIFT Algorithm described in the paper
    Mat output;
    drawKeypoints(grayImg, keypoints, output); // temp
    return output;
}
