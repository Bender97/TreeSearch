//
// Created by cogny on 14/06/20.
//

#ifndef BOF_CLASSIFIERUTILITY_H
#define BOF_CLASSIFIERUTILITY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include "DatasetUtility.h"

using namespace std;
using namespace cv;

double trainModel(string train_dataset_path, Ptr<ml::SVM> svm);
double testModel (string test_dataset_path,  Ptr<ml::SVM> svm);


#endif //BOF_CLASSIFIERUTILITY_H
