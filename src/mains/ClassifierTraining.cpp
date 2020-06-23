//
// Created by cogny on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/ClassifierUtility.h"

int main(int argc, char ** argv) {

    cv::String keys =
            "{train  |../data/csv_dataset/hist_dataset_train.csv   | train csv dataset path}"
            "{svm   |../data/model/svmModel3Chi2.yml       | svm model path}"
            "{help h usage ?    |      | usage: train=train_path svm=svm_path\n"
            "if path contains a space, you should embrace path in quotes}";

    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    string train_path = parser.get<cv::String>("train");
    string svm_path = parser.get<cv::String>("svm");

    Ptr<ml::SVM> svm;

    cout << "Training started" << endl;

    cout << "Training error of the SVM model: " << trainModel(train_path, svm) << endl;
    writeSVMModel(svm_path, svm);
    cout << "SVM model stored at " << svm_path << endl;
    return 0;
}