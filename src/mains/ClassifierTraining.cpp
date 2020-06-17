//
// Created by cogny on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/ClassifierUtility.h"

int main(int args, char ** argv) {

    if (args<3) {
        cout << "usage: train_path svm_path";
        return 1;
    }

    string train_path = argv[1];
    string svm_path = argv[2];

    Ptr<ml::SVM> svm;
    cout << trainModel(train_path, svm) << endl;
    cout << testModel("/home/fusy/Scrivania/Universita/Computer_Vision_1920/Code/BoF/cmake-build-debug/hist_dataset_test.csv", svm) << endl;
    writeSVMModel(svm_path, svm);
    return 0;
}