//
// Created by fusy on 13/06/20.
//

#ifndef BOF_IMAGESUTILITY_H
#define BOF_IMAGESUTILITY_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class ImagesUtility {
public:
    static vector<string> loadImagesPaths(string path);
    static Mat loadImage(string path);
    static bool storeImage(string path, Mat img);
};


#endif //BOF_IMAGESUTILITY_H
