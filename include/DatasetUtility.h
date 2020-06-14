//
// Created by cogny on 14/06/20.
//

#ifndef BOF_DATASETUTILITY_H
#define BOF_DATASETUTILITY_H

#include <iostream>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "FilesUtility.h"
#include "ImagesUtility.h"
#include "ClassifierUtility.h"

#define TRAINING_SET_DIR "hist_dataset_train.csv"
#define TEST_SET_DIR "hist_dataset_test.csv"

#define TREE_DIR "/tree"
#define NON_TREE_DIR "/non_tree"

using namespace std;
using namespace cv;

void buildTrainingSet(string input_images_path, Mat vocabulary, string output_CSVs_path, float proportion);
pair<Mat, Mat> loadDataset(string dataset_path);

#endif //BOF_DATASETUTILITY_H
