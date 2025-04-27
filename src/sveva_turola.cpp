#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "./../include/sveva_turola.hpp"
#include "./../include/marco_annunziata.hpp"

using namespace std;
using namespace cv;

cv::RNG rng(12345);

vector<Point> featureMatching(Mat* img, ObjModel& models){
    Mat image = img->clone();
    vector<KeyPoint> imageKeypoints = SIFT_PCA::detectKeypoints(image);
    Mat imageDescriptors = SIFT_PCA::computeDescriptors(image, imageKeypoints);
    size_t bestNumberMatches = 0;
    int modelIndex = -1;
    vector<DMatch> bestMatch;
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_L2);
    setViewsKeypoints(models);
    setViewsDescriptors(models);

    for(size_t i = 0; i < models.views.size(); i++){
        modelView& view = models.views[i];
        vector<vector<DMatch>> matches;
        matcher -> knnMatch(imageDescriptors, view.descriptors, matches, 2);

        const float ratioTresh = 0.7f;
        vector<DMatch> goodMatches;
        for(size_t j = 0; j < matches.size(); j++){
            if(matches[j][0].distance < ratioTresh * matches[j][1].distance){
                goodMatches.push_back(matches[j][0]);
            }
        }

        if(goodMatches.size() > bestNumberMatches){
            bestNumberMatches = goodMatches.size();
            modelIndex = i;
            bestMatch = goodMatches;
        }
    }

    vector<Point2f> objectPoints;
    vector<Point2f> scenePoints;

    for(size_t i = 0; i < bestMatch.size(); i++){
        objectPoints.push_back(models.views[modelIndex].keypoints[bestMatch[i].trainIdx].pt);
        scenePoints.push_back(imageKeypoints[bestMatch[i].queryIdx].pt);
    }

    Mat homography = findHomography(objectPoints, scenePoints, RANSAC);

    Mat imageMatches;
    drawMatches(image, imageKeypoints, models.views[modelIndex].image, models.views[modelIndex].keypoints, bestMatch, imageMatches, Scalar::all(-1),
    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    Mat canny;
    Canny(image, canny, 100, 200);
    
    vector<vector<Point>> contours;
    findContours(canny, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    vector<Point2f>centers( contours.size() );
    vector<float>radius( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
    {
        approxPolyDP( contours[i], contours_poly[i], 3, true );
        boundRect[i] = boundingRect( contours_poly[i] );
    }
    
    int largestContourIdx = -1;
    double maxArea = 0;

    for (size_t i = 0; i < boundRect.size(); i++) {
        double area = boundRect[i].width * boundRect[i].height;
        if (area > maxArea) {
            maxArea = area;
            largestContourIdx = i;
        }
    }

    if (largestContourIdx != -1) {
        Scalar color = Scalar(0, 255, 0);
        rectangle(imageMatches, boundRect[largestContourIdx].tl(), boundRect[largestContourIdx].br(), color, 2);
    }
    
    imshow("Matches", imageMatches);
    waitKey(0);

    vector<Point> points = {boundRect[largestContourIdx].tl(), boundRect[largestContourIdx].br()};
    return points;

}