//
// Created by fusy on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/VocabularyUtility.h"

int main(int args, char ** argv) {
    cout << "Hello World" << endl;

    storeVocabulary("../vocabulary_dataset/", "vocabulary_200.yml", 200);

    return 0;
}