//
// Created by cogny on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/ClassifierUtility.h"

int main(int argc, char ** argv) {
    cv::String keys =
            "{test  |../data/csv_dataset/hist_dataset_test.csv   | test csv dataset path}"
            "{svm   |../data/model/svmModel3Chi2.yml       | svm model path}"
            "{help h usage ?    |      | usage: test=test_path svm=svm_path\n"
            "if path contains a space, you should embrace path in quotes}";

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    string test_path = parser.get<cv::String>("test");
    string svm_path = parser.get<cv::String>("svm");

    cout << "Loading SVM model: " << svm_path << endl;
    Ptr<ml::SVM> svm = ml::SVM::load(svm_path);
    if (svm->empty() || !svm->isTrained()) {
        cout << "problem loading svmModel provided with path:" << endl << svm_path << endl;
        return 1;
    }
    cout << "SVM model correctly loaded" << endl << endl;

    cout << "Test error of the SVM model: " << testModel(test_path, svm) << endl;

    return 0;
}