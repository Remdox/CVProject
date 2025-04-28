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

vector<Point> featureMatching(Mat* img, ObjModel& models){
    Mat image = img->clone();

    // set keypoints and descriptors of test's image and model
    vector<KeyPoint> imageKeypoints = SIFTDetector::detectKeypoints(image);
    Mat imageDescriptors = SIFTDetector::computeDescriptors(image, imageKeypoints);
    setViewsKeypoints(models);
    setViewsDescriptors(models);

    size_t bestNumberMatches = 0;
    int modelIndex = -1;
    vector<DMatch> bestMatches;

    // creation of the brute force matcher
    Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_L2);

    for(size_t i = 0; i < models.views.size(); i++){
        modelView& view = models.views[i];
        vector<vector<DMatch>> matches;
        matcher -> knnMatch(view.descriptors, imageDescriptors, matches, 2);

        // Lowe's ratio test
        const float ratioTresh = 0.7f;
        vector<DMatch> goodMatches;
        for(size_t j = 0; j < matches.size(); j++){
            if(matches[j][0].distance < ratioTresh * matches[j][1].distance){
                goodMatches.push_back(matches[j][0]);
            }
        }

        // find the best model
        if(goodMatches.size() > bestNumberMatches){
            bestNumberMatches = goodMatches.size();
            modelIndex = static_cast<int>(i);
            bestMatches = goodMatches;
        }
    }

    // find Homography
    vector<Point2f> objectPoints;
    vector<Point2f> scenePoints;
    vector<uchar> maskInliers;
    Mat homography;

    if(bestMatches.size() >= 4){
        for(size_t i = 0; i < bestMatches.size(); i++){
            objectPoints.push_back(models.views[modelIndex].keypoints[bestMatches[i].queryIdx].pt);
            scenePoints.push_back(imageKeypoints[bestMatches[i].trainIdx].pt);
        }

        homography = findHomography(objectPoints, scenePoints, RANSAC, 10.0, maskInliers);
        if (homography.empty()) {
            cout << "Homography is empty!\n";
        }
    } else {
        cout << "Not enough matches found: " << bestMatches.size() << "\n";
        return {Point(INT_MIN, INT_MIN), Point(INT_MIN, INT_MIN)};
    }

    Mat imageMatches;
    vector<char> mask_char(maskInliers.begin(), maskInliers.end());
    drawMatches(models.views[modelIndex].image, models.views[modelIndex].keypoints, image, imageKeypoints, bestMatches, imageMatches, Scalar::all(-1),
    Scalar::all(-1), mask_char, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    // draw bounding box based on keypoints of the best model
    vector<KeyPoint> keypoints = models.views[modelIndex].keypoints;

    Point2f topLeft = keypoints[0].pt;
    Point2f bottomRight = keypoints[0].pt;
    for (size_t i = 1; i < keypoints.size(); i++) {
        const Point2f& pt = keypoints[i].pt;
        if (pt.x < topLeft.x) topLeft.x = pt.x;
        if (pt.y < topLeft.y) topLeft.y = pt.y;
        if (pt.x > bottomRight.x) bottomRight.x = pt.x;
        if (pt.y > bottomRight.y) bottomRight.y = pt.y;
    }

    vector<Point2f> modelCorners = {
        topLeft,
        Point2f(bottomRight.x, topLeft.y),
        bottomRight,
        Point2f(topLeft.x, bottomRight.y)
    };

    vector<Point2f> sceneCorners;
    perspectiveTransform(modelCorners, sceneCorners, homography);

    Size modelSize = models.views[modelIndex].image.size();
    int modelWidth = modelSize.width;

    Point2f modelOffset(modelWidth, 0);

    for (int i = 0; i < 4; i++) {
        line(imageMatches, sceneCorners[i] + modelOffset, sceneCorners[(i + 1) % 4] + modelOffset, Scalar(0, 255, 0), 2);
    }

    imshow("Matches and Bounding Box", imageMatches);
    waitKey(0);

    vector<Point> points = {boundingRect(sceneCorners).tl(), boundingRect(sceneCorners).br()};
    return points;
}
