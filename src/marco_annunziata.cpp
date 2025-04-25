
#include "marco_annunziata.hpp"
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

void getForegroundMask(){

}

Mat pcaSift(Mat* img, vector<Mat> models){
    Mat grayImg = img->clone();
    cvtColor(*img, grayImg, COLOR_BGR2GRAY);
    Ptr<SIFT> sift = SIFT::create(0, 3, 0.04, 10, 1.6, false);
    vector<KeyPoint> keypoints;
    sift.get()->detect(grayImg, keypoints, noArray());

    sift.get()->compute(grayImg, keypoints, noArray()); // TODO (WIP): sub with the real PCA-SIFT Algorithm described in the paper
    Mat output;
    drawKeypoints(grayImg, keypoints, output); // temp
    imwrite("../output/keypoints.png", output);

    return output;
}
