//
// Created by fusy on 13/06/20.
//

#include "FilesUtility.h"

/*Mat FilesUtility::loadYAML(string path, string key) {
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

bool FilesUtility::writeYAML(string path, string key, Mat table) {
    FileStorage fs(path, FileStorage::WRITE);
    fs << key << table;
    fs.release();
    return true;
}

ofstream FilesUtility::prepareCSV(string path) {
    ofstream file;
    file.open(path);
    return file;
}

bool FilesUtility::addRowCSV(ofstream file, Mat histogram, int class_) {
    for (int bin = 0; bin < histogram.cols; bin++) {
        float value = histogram.at<float>(0, bin);
        file << value << ",";
    }

    file << class_ << endl;
    return true;
}*/

/*vector<pair<Mat, int>> FilesUtility::readCSV(string path) {
    ifstream fs;
    fs.open(path);
    // Make sure the file is open
    if(!fs.is_open()) throw std::runtime_error("Could not open file");

    string row ="";

    vector<pair<Mat, int>> result;

    while(getline(fs, row)) {
        //cout << row << endl;
        int init=0, end=0;
        vector<float> hist;
        for(int i=0; i<row.length(); i++) {
            if (row[i] == ',') {
                end = i;

//                cout << "init: " << init << " end: " << end << " i: " << i << endl;
                string sub="";
                for (int c=init; c<end; c++) sub+=row[c];
                float val = stof(sub);
                hist.push_back(val);
                init = end+1;
                i++;
            }

        }

        Mat histogram(Size(hist.size(), 1), CV_16F);
        int size = hist.size()-1;
        for (int i=0; i<size; i++) {
            histogram.at<float>(i) = hist[i];
        }

        string sub="";
        for (int c=init; c<row.length(); c++) sub+=row[c];
        int class_ = stoi(sub);

        cout << class_ << endl;

//        result.push_back(pair<Mat, int>(histogram, class_));
        //result.push_back(pair<Mat, int>(Mat(), class_));
    }
    return result;
}*/

vector<pair<Mat, int>> FilesUtility::readCSV(string path) {
    ifstream fs;
    fs.open(path);
    // Make sure the file is open
    if(!fs.is_open()) throw std::runtime_error("Could not open file");

    string row ="";

    vector<pair<Mat, int>> result;

    while(getline(fs, row)) {
        //cout << row << endl;
        int init=0, end=0;
        vector<float> hist;
        for(int i=0; i<row.length(); i++) {
            if (row[i] == ',') {
                end = i;

//                cout << "init: " << init << " end: " << end << " i: " << i << endl;
                string sub="";
                for (int c=init; c<end; c++) sub+=row[c];
                float val = stof(sub);
                hist.push_back(val);
                init = end+1;
                i++;
            }

        }

        Mat histogram(Size(hist.size(), 1), CV_16F);
        int size = hist.size()-1;
        for (int i=0; i<size; i++) {
            histogram.at<float>(i) = hist[i];
        }

        string sub="";
        for (int c=init; c<row.length(); c++) sub+=row[c];
        int class_ = stoi(sub);

        cout << class_ << endl;

//        result.push_back(pair<Mat, int>(histogram, class_));
        //result.push_back(pair<Mat, int>(Mat(), class_));
    }
    return result;
}