//
// Created by fusy on 14/06/20.
//

#ifndef BOF_VOCABULARYUTILITY_H
#define BOF_VOCABULARYUTILITY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "../include/ImagesUtility.h"
#include "../include/FilesUtility.h"

using namespace std;
using namespace cv;

#define DEFAULT_BAGS 200


bool storeVocabulary(string path_imgs, string voc, int n_bags);
bool storeVocabulary(string path_imgs, string voc);
Mat makeVocabulary(string path_imgs, int n_bags);
Mat makeVocabulary(string path_imgs);


#endif //BOF_VOCABULARYUTILITY_H