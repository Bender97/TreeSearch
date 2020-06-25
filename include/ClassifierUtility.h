//
// Created by cogny on 14/06/20.
//

#ifndef BOF_CLASSIFIERUTILITY_H
#define BOF_CLASSIFIERUTILITY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include "DatasetUtility.h"

#define NON_TREE_CLASS 0
#define TREE_CLASS 1
#define MAYBE_TREE_CLASS 2

using namespace std;
using namespace cv;

/**
 * @brief train a SVM model
 * @param train_dataset_path Path to the folder containing the dataset of images to train on
 * @param svm Class template instantiation reference of the model which will be trained
 * @return accuracy on train dataset
 */
double trainModel(const string &train_dataset_path, Ptr<ml::ANN_MLP> & svm);

/**
 * @brief compute the test error of a SVM model
 * @param test_dataset_path Path to the folder containing the dataset of images to test on
 * @param svm Class template instantiation reference of the model (already trained!)
 * @return accuracy on test dataset
 */
double testModel (const string & test_dataset_path,  Ptr<ml::SVM> svm);


#endif //BOF_CLASSIFIERUTILITY_H
