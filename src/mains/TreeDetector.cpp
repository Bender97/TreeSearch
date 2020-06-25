//
// Created by fusy on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/VocabularyUtility.h"
#include "../../include/Detector.h"

int main(int argc, char ** argv) {

    cv::String keys =
            "{voc   |../data/vocabulary/vocabulary_200.yml  | vocabulary path}"
            "{svm   |../data/model/svmModel.yml   | svm model path}"
            "{img   |\"../data/Benchmark_step_1/Figure 1.jpg\"       | img to analyze path}"
            "{v     |false       | verbosity}"
            "{help h usage ?    |      | usage: -voc=vocabulary_path -svm=svm_path -img=img_path\n"
            "if path contains a space, you should embrace path in quotes}";

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    string vocabulary_path = parser.get<cv::String>("voc");
    string svm_path = parser.get<cv::String>("svm");
    string img_path = parser.get<cv::String>("img" );
    bool verbose = parser.get<bool>("v" );


    if (verbose) cout << "Loading vocabulary: " << vocabulary_path << endl;
    Detector detector;
    detector.setVocabulary(loadYAML(vocabulary_path, DEFAULT_KEY));
    if (verbose) cout << "Vocabulary correctly loaded" << endl << endl;


    if (verbose)cout << "Loading SVM model: " << svm_path << endl;
    Ptr<ml::SVM> svm = ml::SVM::load(svm_path);
//    Ptr<ml::ANN_MLP> svm = ml::ANN_MLP::load("../data/model/ann.yml");

    if (svm->empty() || !svm->isTrained()) {
        cout << "problem loading svmModel provided with path:" << endl << svm_path << endl;
        return 1;
    }

    detector.setClassifier(svm);
    if (verbose) cout << "SVM model correctly loaded" << endl << endl;

    if (verbose) cout << "Loading image: " << img_path << endl;
    Mat img = loadImage(img_path);

    if (img.empty()) {
        cout << "error loading image" << endl;
        return 1;
    }
    if (verbose) cout << "Image correctly loaded" << endl << endl;

    cout << "Starting tree detection" << endl;

    Mat result = detector.detectTrees(img, verbose);

    imshow("result", result);
    waitKey(0);
    return 0;
}