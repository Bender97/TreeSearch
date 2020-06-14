//
// Created by fusy on 13/06/20.
//

#include "../include/FilesUtility.h"

Mat loadYAML(string path, string key) {
    Mat obj;
    FileStorage fs(path, FileStorage::READ);
    if (!fs.isOpened()) {
        // ERROR handling
        return Mat();
    }
    fs[key] >> obj;
    fs.release();
    return obj;
}

bool writeYAML(string path, string key, Mat table) {
    FileStorage fs(path, FileStorage::WRITE);
    fs << key << table;
    fs.release();
    return true;
}

ofstream prepareCSV(string path) {
    ofstream file;
    file.open(path);
    return file;
}

bool addRowCSV(ofstream& file, Mat histogram, int class_) {
    for (int bin = 0; bin < histogram.cols; bin++) {
        float value = histogram.at<float>(0, bin);
        file << value << ",";
    }

    file << class_ << endl;
    return true;
}

vector<pair<Mat, int>> readCSV(string path) {
    ifstream fs;
    fs.open(path);

    // Make sure the file is open
    if (!fs.is_open()) throw std::runtime_error("Could not open file");

    // each row will be read inside this variable
    string row;
    vector<pair<Mat, int>> result;

    // read each row of the CSV
    while (fs >> row) {
        // stream the row and parse it
        stringstream ss(row);
        string token;   // each cell will be saved here

        // container for reading both histogram and class
        vector<float> hist;
        while (std::getline(ss, token, ','))
            hist.push_back(stof(token));

        Mat histogram(Size(hist.size() - 1, 1), CV_32F);

        // fill the Mat with the histogram values (ignore the last element = class_)
        for (int i = 0; i < hist.size() - 1; i++)
            histogram.at<float>(i) = hist[i];

        // extract the class_ value (the last in the vector hist) and cast to int
        int class_ = (int)(hist[hist.size() - 1]);

        result.push_back(pair<Mat, int>(histogram, class_));
    }
    return result;
}

Ptr<ml::SVM> loadSVMModel(string path) {
    Ptr<ml::SVM> svm;
    svm->load(path);
    return svm;
}

void writeSVMModel(string path, Ptr<ml::SVM> svm)
{
    svm->save(path);
}

