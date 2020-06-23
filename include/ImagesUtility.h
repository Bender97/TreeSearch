//
// Created by fusy on 13/06/20.
//

#ifndef BOF_IMAGESUTILITY_H
#define BOF_IMAGESUTILITY_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**
 * @brief load the paths of the images contained in a folder
 * @param path Path of the folder
 * @return vector of strings containing the filenames of all the images
 */
vector<string> loadImagesPaths(string path);

/**
 * @brief load an image using opencv primitives
 * @param path Path to the image
 * @return Mat variable containing the loaded image
 */
Mat loadImage(string path);

/**
 * @brief store an image using opencv primitives
 * @param path Path where to store the image
 * @param img Img to store
 * @return true if no errors occurred, false otherwise
 */
bool storeImage(string path, Mat img);

/**
 * @brief compute the windowing in desired rows and cols of an image
 * @param img Img to compute windowing of
 * @param rows Number of rows
 * @param cols Number of cols
 * @return computed windows
 */
vector<Rect> getWindows(Mat img, int rows, int cols);

/**
 * @brief compute the windowing in desired rows and cols of an image
 *          starting from the (x, y) position
 * @param rows Number of rows
 * @param cols Number of cols
 * @param x X starting position (top-left)
 * @param y Y starting position (top-left)
 * @param w_size Window size, inside which to compute windowing
 * @return computed windows
 */
vector<Rect> getFrames(int rows, int cols, int x, int y, int w_size);

#endif //BOF_IMAGESUTILITY_H
