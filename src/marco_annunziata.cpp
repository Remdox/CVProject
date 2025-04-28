
#include "marco_annunziata.hpp"
#include "shared.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <numeric>

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
        view.keypoints = SIFTDetector::detectKeypoints(view.image, view.mask);
    }
}

void setViewsDescriptors(ObjModel& model){
    for(modelView& view : model.views){
        view.descriptors = SIFTDetector::computeDescriptors(view.image, view.keypoints);
    }
}

double computeOverlap(const Mat& mask1, const Mat& mask2) {
    Mat intersection;
    bitwise_and(mask1, mask2, intersection);
    int pixelsIntersection = countNonZero(intersection);
    int pixelsUnion = countNonZero(mask1) + countNonZero(mask2) - pixelsIntersection;

    return (pixelsUnion == 0) ? 0.0 : static_cast<double>(pixelsIntersection) / pixelsUnion;
}

Mat meanShiftSegmentation(Mat &src){
    Mat segmented;
    pyrMeanShiftFiltering(src, segmented, 40, 60);

    Mat processedMask = Mat::zeros(src.size(), CV_8UC1);
    int segmentNumber = 0;
    vector<Mat> existingSegments;


    for(int y = 0; y < segmented.rows; y++) {
        for(int x = 0; x < segmented.cols; x++) {
            if(processedMask.at<uchar>(y, x) == 0) {
                Mat floodMask = Mat::zeros(segmented.rows + 2, segmented.cols + 2, CV_8UC1);
                Rect rect;

                int flags = 8 | FLOODFILL_MASK_ONLY | FLOODFILL_FIXED_RANGE;
                floodFill(segmented, floodMask, Point(x, y), Scalar(), &rect,
                         Scalar(40, 40, 40), Scalar(40, 40, 40), flags);

                Mat segmentMask = floodMask(Rect(1, 1, segmented.cols, segmented.rows));
                segmentMask.convertTo(segmentMask, CV_8UC1, 255);

                if(countNonZero(segmentMask) > 500) {
                    Mat segment;
                    src.copyTo(segment, segmentMask);

                    bool isDuplicate = false;

                    for (const auto& existing : existingSegments) {
                        double overlap = computeOverlap(existing, segmentMask);
                            if (overlap > 0.3) { // Soglia di sovrapposizione (30%)
                                isDuplicate = true;
                                break;
                            }
                    }

                    if (!isDuplicate) {
                        existingSegments.push_back(segmentMask.clone());
                        segmentNumber++;
                    }
                }

                bitwise_or(processedMask, segmentMask, processedMask);

                Mat kernel_open = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
                Mat kernel_close = getStructuringElement(MORPH_ELLIPSE, Size(11,11));


                // Applica apertura per rimuovere il rumore
                morphologyEx(processedMask, processedMask, MORPH_OPEN, kernel_open);
                // Poi applica chiusura per unire le regioni
                morphologyEx(processedMask, processedMask, MORPH_CLOSE, kernel_close);

                /*Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
                morphologyEx(processedMask, processedMask, MORPH_CLOSE, kernel);*/
            }
        }
    }

    vector<int> segmentSize;
    for(const auto& segment : existingSegments){
        segmentSize.push_back(countNonZero(segment));
    }

    vector<int> segmentsIndices(existingSegments.size());
    iota(segmentsIndices.begin(), segmentsIndices.end(), 0);
    sort(segmentsIndices.begin(), segmentsIndices.end(), [&segmentSize](int a, int b) {
        return segmentSize[a] > segmentSize[b];
    });

    Mat mergedMask = Mat::zeros(src.size(), CV_8UC1);
    int numSegToMerge = min(3, (int)existingSegments.size());
    for(int i = 0; i < numSegToMerge; i++) {
        bitwise_or(mergedMask, existingSegments[segmentsIndices[i]], mergedMask);
    }

    Mat risultato;
    src.copyTo(risultato, mergedMask);

    cout << segmentNumber << " segments found!" << endl;
    return mergedMask;
}

Mat segmentImgBackground(Mat &src){
    Mat segmentedBg = meanShiftSegmentation(src);
    Mat result = Mat::zeros(src.size(), src.type());;
    Mat mask;

    // inverting black and non-black pixels of the segmented background to get a mask to use for SIFT
    // To do this, I create and apply a mask to the segmented bg
    compare(segmentedBg, Scalar(0), mask, CMP_EQ);
    src.copyTo(result, mask);
    imwrite("output/backgroundSegmentation.png", result);

    return result;
}


// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@ SIFTDetector Class @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

// static variabile definition
Ptr<SIFT> SIFTDetector::sift = SIFT::create(0,
                                        3,
                                        0.04,
                                        10,
                                        1.6,
                                        false);

// static methods
vector<KeyPoint> SIFTDetector::detectKeypoints_canny(Mat& img) {
    Mat grayImg, edges, mask;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    Canny(grayImg, edges, 50, 150);
    dilate(edges, mask, Mat(), Point(-1,-1), 3);

    vector<KeyPoint> keypoints;
    sift->detect(grayImg, keypoints, mask);
    return keypoints;
}

// to use for the testImg (I actually could merge them into one, but that would break the code of my collegues and I don't want to waste their time changing their code too many times)
vector<KeyPoint> SIFTDetector::detectKeypoints(Mat& img){
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    Mat mask = segmentImgBackground(img);
    cvtColor(mask, mask, COLOR_BGR2GRAY);
    threshold(mask, mask, 1, 255, THRESH_BINARY); // Forza valori binari

    vector<KeyPoint> keypoints;
    sift->detect(grayImg, keypoints, mask);

    return keypoints;
}

// to use for the models
vector<KeyPoint> SIFTDetector::detectKeypoints(Mat& img, Mat& mask) {
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    Mat grayMask;
    cvtColor(mask, grayMask, COLOR_BGR2GRAY);
    threshold(grayMask, grayMask, 1, 255, THRESH_BINARY); // Forza valori binari

    vector<KeyPoint> keypoints;
    sift->detect(grayImg, keypoints, grayMask);

    return keypoints;
}

Mat SIFTDetector::computeDescriptors(Mat& img, vector<KeyPoint> keypoints){
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    Mat descriptors;
    // TODO (WIP): sub with the real PCA-SIFT Algorithm described in the paper
    sift->compute(grayImg, keypoints, descriptors);
    return descriptors;
}

