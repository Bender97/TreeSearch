//
// Created by fusy on 14/06/20.
//

#ifndef BOF_VOCABULARYUTILITY_H
#define BOF_VOCABULARYUTILITY_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "../include/ImagesUtility.h"
#include "../include/FilesUtility.h"

#define DEFAULT_KEY "vocabulary"
#define DEFAULT_BAGS 2000

using namespace std;
using namespace cv;

/**
 * @brief store a vocabulary in YAML format
 * @param path_imgs Path to the dataset of images to compute the vocabulary
 * @param path_voc Path of the file to use to store the vocabulary
 * @param n_bags Number of bags (bins of the future histogram)
 * @return true if no errors occurred, false otherwise
 */
bool storeVocabulary(string path_imgs, string path_voc, int n_bags);
bool storeVocabulary(string path_imgs, string path_voc);

/**
 * @brief build a vocabulary from a dataset of images
 * @param path_imgs Path to the dataset of images
 * @param n_bags Number of bags (bins of the histogram)
 * @return Mat variable containing the vocabulary (each row is a descriptor)
 */
Mat makeVocabulary(string path_imgs, int n_bags);
Mat makeVocabulary(string path_imgs);


#endif //BOF_VOCABULARYUTILITY_H
