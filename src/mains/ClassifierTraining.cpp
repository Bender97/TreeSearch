//
// Created by cogny on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/ClassifierUtility.h"

int main(int args, char ** argv) {
    cout << "Hello World" << endl;
    Ptr<ml::SVM> svm;
    cout << trainModel("/home/fusy/Scrivania/Universita/Computer_Vision_1920/Code/BoF/cmake-build-debug/hist_dataset_train.csv", svm) << endl;
    cout << testModel("/home/fusy/Scrivania/Universita/Computer_Vision_1920/Code/BoF/cmake-build-debug/hist_dataset_test.csv", svm) << endl;
    writeSVMModel("/home/fusy/Scrivania/Universita/Computer_Vision_1920/Code/BoF/cmake-build-debug/svmModel.yml", svm);
    return 0;
}