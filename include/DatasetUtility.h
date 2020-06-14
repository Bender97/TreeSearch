//
// Created by cogny on 14/06/20.
//

#ifndef BOF_DATASETUTILITY_H
#define BOF_DATASETUTILITY_H

#include <iostream>
#include <opencv2/opencv.hpp>

#include "FilesUtility.h"
#include "ImagesUtility.h"

using namespace std;
using namespace cv;

void buildTrainingSet(string input_images_path, Mat vocabulary, string output_images_path, float proportion);
Mat loadDataset(string dataset_path);

#endif //BOF_DATASETUTILITY_H
