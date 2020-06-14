//
// Created by cogny on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/VocabularyUtility.h"
#include "../../include/DatasetUtility.h"

int main(int args, char ** argv)
{
    
    string dataset_images_path = "../../train_dataset_images/";
    string vocabulary_path = "../../cmake-build-debug/vocabulary.yml";
    string dataset_csv_path = "../../cmake-build-debug/";
    float proportion = 0.8;

    Mat vocabulary = loadYAML(vocabulary_path, DEFAULT_KEY);

    buildTrainingSet(dataset_images_path, vocabulary, dataset_csv_path, proportion);

    return 0;
}