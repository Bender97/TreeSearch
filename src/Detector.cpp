//
// Created by fusy on 14/06/20.
//

#include "../include/Detector.h"

void Detector::setVocabulary(Mat vocabulary) {
    this->vocabulary = vocabulary;
}

void Detector::setClassifier(Ptr<ml::SVM> classifier) {
    this->classifier = classifier;
}

Mat Detector::detectTrees(Mat img) {
    //create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create Sift feature point detector
    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(detector, matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(this->vocabulary);

    cvtColor(img, img, COLOR_BGR2GRAY);
    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;

    int min_size = min(img.rows, img.cols);
    int step = 50;

    for (int scale = 1; scale < 5; scale++) {
        int w_size = min_size / scale;

        for (int x = 0; x <= img.cols - w_size; x += step) {
            for (int y = 0; y <= img.rows - w_size; y += step) {
                Rect window(x, y, w_size, w_size);

                //Detect SIFT keypoints (or feature points)
                detector->detect(img(window), keypoints);
                //To store the BoW (or BoF) representation of the image
                Mat histogram;
                //extract BoW (or BoF) descriptor from given image

                bowDE.compute(img, keypoints, histogram);

                int response = (int) this->classifier->predict(histogram);

                if (response == 1)
                    rectangle(img, window, { 0, 0, 255 });
            }
        }


    }

    return img;
}
