//
// Created by fusy on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/VocabularyUtility.h"
#include "../../include/FilesUtility.h"
#include "../../include/Detector.h"

int main(int args, char ** argv) {
    cout << "Hello World" << endl;
    Detector conan;
    conan.setVocabulary(loadYAML("/home/fusy/Scrivania/Universita/Computer_Vision_1920/Code/BoF/cmake-build-debug/vocabulary_200.yml", DEFAULT_KEY));

    Ptr<ml::SVM> svm = ml::SVM::load("/home/fusy/Scrivania/Universita/Computer_Vision_1920/Code/BoF/cmake-build-debug/svmModel.yml");
    cout << svm->isTrained() << endl;
    conan.setClassifier(svm);
    Mat img = loadImage("/home/fusy/Scrivania/Universita/Computer_Vision_1920/Code/BoF/Benchmark_step_1/Figure 1.jpg");
    while (img.cols > 800 || img.rows > 600)
        resize(img, img, Size(img.cols / 2, img.rows / 2));
    Mat res = conan.detectTrees(img);

    imshow("res", res);
    waitKey(0);
    return 0;
}