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
namespace fs = std::filesystem;

// we can use everything in OpenCV except Deep Learning, we can use ML and can explore
// something different from what we've seen the lectures

int alternativeMain();

int main(int argc, char** argv){
    //Usato per fare il batch
    return alternativeMain();
    if(argc < 3){
        cerr << "Usage: <test image path> <object_detection_dataset path>\n";
        return -1;
    }


    const string WINDOWNAME = "Test image";
    // string imgPath     = argv[1];
    // string datasetPath = argv[2];
    string imgPath     = "../data/object_detection_dataset/006_mustard_bottle/test_images/6_0001_000952-color.jpg";
    string datasetPath = "../data/object_detection_dataset/";
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
    // namedWindow(WINDOWNAME);
    // imshow(WINDOWNAME, testImg);

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

    // ... (dove hai già popolato sugarPoints, mustardPoints e drillPoints)
    Point sugarTl    = sugarPoints[0];
    Point sugarBr    = sugarPoints[1];
    Point mustardTl  = mustardPoints[0];
    Point mustardBr  = mustardPoints[1];
    Point drillTl    = drillPoints[0];
    Point drillBr    = drillPoints[1];

    // Costruisci i Rect
    Rect rectSugar   = makeRect(sugarTl.x,    sugarTl.y,    sugarBr.x,    sugarBr.y);
    Rect rectMustard = makeRect(mustardTl.x,  mustardTl.y,  mustardBr.x,  mustardBr.y);
    Rect rectDrill   = makeRect(drillTl.x,    drillTl.y,    drillBr.x,    drillBr.y);

    // Calcola le 3 metriche
    ObjMetric metricSugar   = computeMetrics(imgPath, labelPath, ImgObjType::sugar_box,       rectSugar);
    ObjMetric metricMustard = computeMetrics(imgPath, labelPath, ImgObjType::mustard_bottle, rectMustard);
    ObjMetric metricDrill   = computeMetrics(imgPath, labelPath, ImgObjType::power_drill,    rectDrill);

    // Stampa a video
    std::cout << "Sugar box metric: "    << metricSugar.toString()   << std::endl;
    std::cout << "Mustard bottle metric: "<< metricMustard.toString() << std::endl;
    std::cout << "Power drill metric: "   << metricDrill.toString()   << std::endl;


    //ImgObjType objType = ImgObjType::sugar_box; //Leggiamo un altro param di input ?
    

    //Mat descriptors;

    //Test metrics
    //Dopo aver fatto il matching, passare qua i 2 punti della bounding box
    //Rect rectFound = makeRect(/*xmin, ymin, xmax, ymax*/);
    //Rect rectFound = makeRect(420, 300, 550, 450);
    //ObjMetric metric = computeMetrics(imgPath, labelPath, objType, rectFound);
    //metric.toString();

    // waitKey(0);
    return(0);
}

int alternativeMain() {
    const std::array<ImgObjType,3> objTypes {
        ImgObjType::sugar_box,
        ImgObjType::mustard_bottle,
        ImgObjType::power_drill
    };

    const std::string inputRootFolder  = "./../data/object_detection_dataset/";
    const std::string outputFolderPath = "./../output";

    // --- 1) Pulisci e ricrea output ---
    fs::path outputDir(outputFolderPath);
    if (fs::exists(outputDir)) fs::remove_all(outputDir);
    fs::create_directories(outputDir);

    // --- 2) CARICA TUTTI I MODELLI 1 sola volta ---
    std::vector<ObjModel> models;
    for (auto type : objTypes) {
        ObjModel m;
        m.type = type;
        std::string globPath = (fs::path(inputRootFolder)
                                / getFolderNameData(type)
                                / "models"
                                / "*_color.png").string();
        getModelViews(globPath, m);
        models.push_back(std::move(m));
    }

    // --- 3) Per ciascun tipo, processa la sua cartella di test_images ---
    for (auto type : objTypes) {
        // directory di test per questo tipo
        fs::path testImagesDir = fs::path(inputRootFolder)
                                 / getFolderNameData(type)
                                 / "test_images";
        if (!fs::is_directory(testImagesDir))
            throw std::runtime_error("Manca test_images per " + toString(type));

        // crea sottocartella di output/<type>
        fs::path outSubdir = outputDir / toString(type);
        fs::create_directories(outSubdir);

        // ciclo su ogni immagine di test
        for (auto const& entry : fs::directory_iterator(testImagesDir)) {
            if (!entry.is_regular_file()) continue;
            std::string imgName = entry.path().filename().string();
            if (imgName.size() <= 9 ||
                imgName.substr(imgName.size() - 9) != "color.jpg")
                continue;

            // leggi l’immagine
            cv::Mat testImg = cv::imread(entry.path().string());

            // prepara <base>data.txt
            std::string base = imgName.substr(0, imgName.size() - 9);
            fs::path dataFile = outSubdir / (base + "data.txt");
            std::ofstream ofs(dataFile.string());
            if (!ofs)
                throw std::runtime_error("Impossibile aprire " + dataFile.string());

            // fai matching con TUTTI i modelli (non const!)
            for (auto& model : models) {
                // path dove salvare il bounding‐box
                std::string bbPath = (outSubdir
                                      / (base + "_" + toString(model.type) + "_bb.jpg"))
                                      .string();

                // CHIAMATA CORRETTA: featureMatching richiede ObjModel& mutabile
                std::vector<cv::Point> pts =
                    featureMatching(&testImg, model, bbPath);

                int xmin = pts[0].x, ymin = pts[0].y;
                int xmax = pts[1].x, ymax = pts[1].y;

                // una riga per modello
                ofs << toString(model.type)
                    << " " << xmin
                    << " " << ymin
                    << " " << xmax
                    << " " << ymax
                    << "\n";
            }

            // ofs chiude automaticamente
        }
    }

    return 0;
}
