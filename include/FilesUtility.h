//
// Created by fusy on 13/06/20.
//

#ifndef BOF_FILESUTILITY_H
#define BOF_FILESUTILITY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <sstream>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

Mat loadYAML(string path, string key);
bool writeYAML(string path, string key, Mat table);
vector<pair<Mat, int>> readCSV(string path);
ofstream prepareCSV(string path);
bool addRowCSV(ofstream& file, Mat histogram, int class_);
void loadSVMModel(string path, Ptr<ml::SVM> &svm);
void writeSVMModel(string path, Ptr<ml::SVM> svm);



#endif //BOF_FILESUTILITY_H
