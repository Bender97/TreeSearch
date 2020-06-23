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

/**
 * @brief load the content of a YAML file into a Mat variable
 * @param path Path to the YAML file
 * @param key Key of the desired content to load
 * @return the Mat variable with the file content
 */
Mat loadYAML(string path, string key);

/**
 * @brief store a Mat variable to a YAML file
 * @param path Path to the output file
 * @param key Key to assign to the variable, in order for it to be recognizable
 * @param table Mat variable to store
 * @return true if no errors occurred, false otherwise
 */
bool writeYAML(string path, string key, Mat table);

/**
 * @brief read a csv file containing a dataset containing histograms and true class
 * @param path Path to the csv file
 * @return a vector of pairs such that: first: histogram, second: true class
 */
vector<pair<Mat, int>> readCSV(string path);

/**
 * @brief get the file descriptor of a file in order to store a dataset in csv format
 * @param path Path to the desired file to write onto
 * @return the File Descriptor of the file
 */
ofstream prepareCSV(string path);

/**
 * @brief add a row to a csv file
 * @param file File Descriptor of the out file
 * @param histogram Histogram to store
 * @param class_ True class related to the histogram
 * @return true if no errors occurred, false otherwise
 */
bool addRowCSV(ofstream& file, Mat histogram, int class_);

//void loadSVMModel(string path, Ptr<ml::SVM> &svm);

/**
 * @brief store a SVMModel to a file
 * @param path Path of the YAML file to store the SVM model in
 * @param svm SVM model to store
 */
void writeSVMModel(string path, Ptr<ml::SVM> svm);

#endif //BOF_FILESUTILITY_H
