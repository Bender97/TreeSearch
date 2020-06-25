//
// Created by fusy on 14/06/20.
//

#ifndef BOF_DETECTOR_H
#define BOF_DETECTOR_H


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>
#include "ImagesUtility.h"
#include "ClassifierUtility.h"

using namespace std;
using namespace cv;

class Detector {
public:
    Mat vocabulary;                                 // vocabulary of the BoW
    Ptr<ml::ANN_MLP> classifier;                        // SVM classifier

    /**
     * @brief set the vocabulary to use
     * @param vocabulary BoW vocabulary
     */
    void setVocabulary(Mat vocabulary);

    /**
     * @brief set the classifier model to use
     * @param classifier SVM classifier
     */
    void setClassifier(Ptr<ml::ANN_MLP> &classifier);

    /**
     * @brief look for trees in the img
     * @param img Mat to look for trees
     * @return img with found trees inside a bounding box
     */
    Mat detectTrees(Mat img, bool verbose);

    /**
     * @brief suppress all spammed bounding box to select only suitable ones
     * @param regions Bounding Boxes found
     * @param classes Classes the classifier can recognize
     * @param max_span Scaling factor of the resulting outer region
     * @param score_threshold Threshold to select eligible windows
     * @return final bounding boxes found and filtered
     */
    static vector<Rect> unifyRegions(vector<Rect> regions, vector<int> classes, float max_span, float score_threshold);
    //static vector<Rect> unifyRegionsClustering(vector<Rect> regions, vector<int> classes, float dist_threshold, float score_threshold);

    /**
     * @brief determine if a point is inside a rect
     * @param pt Point to analyze
     * @param rect The rect that may contain the pt
     * @return true if pt is inside rect, false otherwise
     */
    static bool isInsideRect(Point2i pt, Rect rect);

    /**
     * @brief determine if a rect is contained in a rect
     * @param inside Maybe contained rect
     * @param rect Maybe container rect
     * @return true if 'inside' is contained in 'rect', false otherwise
     */
    static bool isInsideRect(Rect inside, Rect rect);
};


#endif //BOF_DETECTOR_H
