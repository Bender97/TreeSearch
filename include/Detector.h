//
// Created by fusy on 14/06/20.
//

#ifndef BOF_DETECTOR_H
#define BOF_DETECTOR_H


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

class Detector {
public:
    Mat vocabulary;
    Ptr<ml::SVM> classifier;
    void setVocabulary(Mat vocabulary);
    void setClassifier(Ptr<ml::SVM> &classifier);
    Mat detectTrees(Mat img);
};


#endif //BOF_DETECTOR_H
