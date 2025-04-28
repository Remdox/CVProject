//includes
#include <opencv2/core/types.hpp>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <array>
#include <filesystem>

//libs
#include "../include/shared.hpp"
#include "../include/marco_annunziata.hpp"
#include "../include/hermann_serain.hpp"
#include "../include/sveva_turola.hpp"

using namespace std;
using namespace cv;
using namespace HermannLib;
using namespace Shared;
//namespace fs = std::filesystem;

// we can use everything in OpenCV except Deep Learning, we can use ML and can explore
// something different from what we've seen the lectures

int alternativeMain();

int main(int argc, char** argv){
    // return alternativeMain();
    if(argc < 3){
        cerr << "Usage: <test image path> <object_detection_dataset path>\n";
        return -1;

    }
    const string WINDOWNAME = "Test image";
    string imgPath     = argv[1];
    string datasetPath = argv[2];
    const string drillViewsPath   = datasetPath + "035_" + toString(ImgObjType::power_drill)    + "/models/*_color.png";
    const string sugarViewsPath   = datasetPath + "004_" + toString(ImgObjType::sugar_box)      + "/models/*_color.png";
    const string mustardViewsPath = datasetPath + "006_" + toString(ImgObjType::mustard_bottle) + "/models/*_color.png";

    string labelPath = imgPath;
    labelPath = labelPath.replace(labelPath.find("test_images"), 11, "labels");
    labelPath.replace(labelPath.find("-color.jpg"), 10, "-box.txt");

    //Mat testImg = imread("../data/object_detection_dataset/006_mustard_bottle/test_images/6_0001_000121-color_2.jpg");
    Mat testImg = imread(imgPath);
    if(testImg.empty() == 1){
        cerr << "No image given as parameter\n";
        return -1;
    }
    namedWindow(WINDOWNAME);
    imshow(WINDOWNAME, testImg);

    ObjModel drillModel;
    ObjModel sugarModel;
    ObjModel mustardModel;
    drillModel.type = ImgObjType::power_drill;
    sugarModel.type = ImgObjType::sugar_box;
    mustardModel.type = ImgObjType::mustard_bottle;
    getModelViews(drillViewsPath, drillModel);
    getModelViews(sugarViewsPath, sugarModel);
    getModelViews(mustardViewsPath, mustardModel);

    vector<string> labels;
    if(getLabels(&labels, labelPath) == -1){
        cerr << "No label file found";
        return -1;
    }

    /*Mat imggtest = imread("../data/object_detection_dataset/006_mustard_bottle/models/view_60_003_color.png");
    //vector<KeyPoint> testImgKpts = SIFT_PfiCA::detectKeypoints(testImg);
    Mat test = imread("../data/object_detection_dataset/006_mustard_bottle/models/view_60_003_mask.png");*/
    vector<KeyPoint> testImgKpts = SIFTDetector::detectKeypoints(testImg);
    Mat outputImg = testImg.clone();
    drawKeypoints(testImg, testImgKpts, outputImg, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite("../output/keypoints.png", outputImg);

    double t = (double)cv::getTickCount();
    cout << "Loading drill models descriptors.." << endl;
    setViewsKeypoints(drillModel);
    cout << "Loading sugar models descriptors.." << endl;
    setViewsKeypoints(sugarModel);
    cout << "Loading mustard models descriptors.." << endl;
    setViewsKeypoints(mustardModel);
    int ind = 0;
    for(modelView view : mustardModel.views){
        outputImg = view.image.clone();
        drawKeypoints(view.image,
                  view.keypoints, outputImg, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string file = "../output/views/view_" + to_string(ind) +  + ".png";
        imwrite(file, outputImg);
        ind++;
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Overall time for feature detection: " << t << " seconds" << std::endl;

    t = (double)cv::getTickCount();
    setViewsDescriptors(drillModel);
    setViewsDescriptors(sugarModel);
    setViewsDescriptors(mustardModel);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Overall time for feature description: " << t << " seconds" << std::endl;

    vector<Point> mustardPoints = featureMatching(&testImg, mustardModel);
    vector<Point> drillPoints = featureMatching(&testImg, drillModel);
    vector<Point> sugarPoints = featureMatching(&testImg, sugarModel);

    /*Point mustardTl = mustardPoints[0];
    Point mustardBr = mustardPoints[1];
    Point drillTl = drillPoints[0];
    Point drillBr = drillPoints[1];
    Point sugarTl = sugarPoints[0];
    Point sugarBr = sugarPoints[1];*/

    //ImgObjType objType = ImgObjType::sugar_box; //Leggiamo un altro param di input ?
    

    //Mat descriptors;

    //Test metrics
    //Dopo aver fatto il matching, passare qua i 2 punti della bounding box
    //Rect rectFound = makeRect(/*xmin, ymin, xmax, ymax*/);
    //Rect rectFound = makeRect(420, 300, 550, 450);
    //ObjMetric metric = computeMetrics(imgPath, labelPath, objType, rectFound);
    //metric.toString();

    waitKey(0);
    return(0);
}

/*
int alternativeMain(){
    const int length = 3;
    std::array<ImgObjType, length> objTypes{
        ImgObjType::sugar_box,
        ImgObjType::mustard_bottle,
        ImgObjType::power_drill
    };

    const std::string inputRootFolder  = "./../data/object_detection_dataset/";
    const std::string outputFolderPath = "./output";

    // 1) reset della cartella output
    fs::path outputDir(outputFolderPath);
    if (fs::exists(outputDir)) {
        fs::remove_all(outputDir);
    }
    fs::create_directories(outputDir);

    // 2) dizionario metriche
    std::map<ImgObjType, std::pair<std::vector<ObjMetric>, double>> metricsDict;

    for (auto const& type : objTypes) {
        // 2a) creazione della sottocartella output/type
        fs::path outSubdir = outputDir / toString(type);
        fs::create_directories(outSubdir);

        // container temporanei per questo tipo
        std::vector<ObjMetric> metricsList;
        double sumIoU = 0.0;

        // percorsi dati
        fs::path dataFolder   = fs::path(inputRootFolder) / getFolderNameData(type);
        fs::path testImagesDir = dataFolder / "test_images";
        fs::path labelsDir     = dataFolder / "labels";

        if (!fs::is_directory(testImagesDir) || !fs::is_directory(labelsDir)) {
            throw std::runtime_error("Directory test_images o labels mancante per " 
                                     + toString(type));
        }

        const std::string suffixImg   = "color.jpg";
        const std::string suffixLabel = "box.txt";

        // 2b) ciclo sulle immagini di test
        for (auto const& testEntry : fs::directory_iterator(testImagesDir)) {
            if (!testEntry.is_regular_file()) 
                continue;

            std::string imgName = testEntry.path().filename().string();
            if (imgName.size() <= suffixImg.size() ||
                imgName.substr(imgName.size() - suffixImg.size()) != suffixImg) {
                continue; // pattern non valido
            }

            // path label corrispondente
            fs::path labelPath = labelsDir 
                               / (imgName.substr(0, imgName.size() - suffixImg.size())
                                  + suffixLabel);

            if (!fs::exists(labelPath)) {
                throw std::runtime_error("Label file mancante: " + labelPath.string());
            }

            // Detection
            std::string testImgPath = testEntry.path().string();
            cv::Mat testImgData     = cv::imread(testImgPath);
            auto keypoints          = SIFTDetector::detectKeypoints(testImgData);

            // Matching …
            // Metricas (finto rect)
            int xmintest = 420, ymintest = 300, xmaxtest = 550, ymaxtest = 500;
            cv::Rect foundRect = makeRect(xmintest, ymintest, xmaxtest, ymaxtest);

            ObjMetric metric = computeMetrics(
                testImgPath, labelPath.string(), type, foundRect
            );

            // raccolta metriche
            metricsList.push_back(metric);
            sumIoU += metric.getIoU();

            // ———————— SCRITTURA FILE “data.txt” ————————
            //SALVARE QUA L'IMMAGINE CON LA BOUNDING BOX
            // genero il nome: "<id>-data.txt"
            std::string dataFileName = 
                imgName.substr(0, imgName.size() - suffixImg.size()) + "data.txt";
            fs::path dataFilePath = outSubdir / dataFileName;

            std::ofstream ofs(dataFilePath);
            if (!ofs) {
                throw std::runtime_error("Impossibile aprire file: " 
                                         + dataFilePath.string());
            }
            // salvo le coordinate del rect
            ofs << xmintest << " " 
                << ymintest << " " 
                << xmaxtest << " " 
                << ymaxtest << "\n";
            ofs.close();
            // ————————————————————————————————
        }

        // 2c) calcolo mIoU e popolo la mappa
        double meanIoU = metricsList.empty()
                         ? 0.0
                         : sumIoU / static_cast<double>(metricsList.size());
        metricsDict[type] = std::make_pair(std::move(metricsList), meanIoU);
    }

    return 0;
}
*/
