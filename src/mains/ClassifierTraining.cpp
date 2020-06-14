//
// Created by cogny on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/ClassifierUtility.h"

int main(int args, char ** argv) {
    cout << "Hello World" << endl;
    Ptr<ml::SVM> svm;
    trainModel("/home/fusy/Scrivania/Universita/Computer_Vision_1920/Code/BoF/dataset.csv", svm);
    return 0;
}