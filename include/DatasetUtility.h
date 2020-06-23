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

#define TREE_DIR "tree/"
#define MAYBE_TREE_DIR "maybe_tree/"
#define NON_TREE_DIR "non_tree/"

using namespace std;
using namespace cv;

/**
 * @brief build 2 csv files, one for training, one for testing
 * @param input_images_path Path to the dataset of images
 * @param vocabulary Vocabulary to use in order to calculate BoW descriptors
 * @param output_CSVs_path Path to the folder where to store the 2 csv files
 * @param proportion Percentage of images to use as train set wrt to images to use as test set
 */
void buildTrainingSet(string input_images_path, Mat vocabulary, string output_CSVs_path, float proportion);

/**
 * @brief load a dataset in the correct format from a csv file
 * @param dataset_path Path to the csv file containing a dataset (either for train or test)
 * @return pair object containing: first: BoW descriptors, second: correct class
 */
pair<Mat, Mat> loadDataset(string dataset_path);

#endif //BOF_DATASETUTILITY_H
