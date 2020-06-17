//
// Created by fusy on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/VocabularyUtility.h"
#include "../../include/Detector.h"

int main(int args, char ** argv) {

    if (args<4) {
        cout << "usage: vocabulary_path svm_path img_path";
        return 1;
    }

    string vocabulary_path = argv[1];
    string svm_path = argv[2];
    string img_path = argv[3];

    Detector detector;
    detector.setVocabulary(loadYAML(vocabulary_path, DEFAULT_KEY));

    Ptr<ml::SVM> svm = ml::SVM::load(svm_path);

    if (svm->empty() || !svm->isTrained()) {
        cout << "problem loading svmModel provided with path:" << endl << svm_path << endl;
        return 1;
    }

    detector.setClassifier(svm);
    Mat img = loadImage(img_path);

    if (img.empty()) {
        cout << "error loading image" << endl;
        return 1;
    }

    /*** RESIZE IMAGE (if needed): faster computation and better visualization ***/
    while (img.cols > 800 || img.rows > 600)
        resize(img, img, Size(img.cols / 2, img.rows / 2));


    Mat res = detector.detectTrees(img);

    imshow("res", res);
    waitKey(0);
    return 0;
}