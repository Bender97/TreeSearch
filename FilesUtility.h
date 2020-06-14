//
// Created by fusy on 13/06/20.
//

#ifndef BOF_FILESUTILITY_H
#define BOF_FILESUTILITY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

class FilesUtility {
public:
    /*static Mat loadYAML(string path, string key);
    static bool writeYAML(string path, string key, Mat table);
    static ofstream prepareCSV(string path);
    static bool addRowCSV(ofstream file, Mat histogram, int class_);*/
    static vector<pair<Mat, int>> readCSV(string path);
//    static Ptr<ml::SVM> loadSVMModel(string path);
};


#endif //BOF_FILESUTILITY_H
