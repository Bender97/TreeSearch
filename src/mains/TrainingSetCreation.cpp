//
// Created by cogny on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/VocabularyUtility.h"
#include "../../include/DatasetUtility.h"

int main(int args, char ** argv)
{
    if (args<5) {
        cout << "usage: dataset_images_path vocabulary_path dataset_csv_path proportion";
        return 1;
    }

    string dataset_images_path = argv[1];
    string vocabulary_path = argv[2];
    string dataset_csv_path = argv[3];

    float proportion = stof(argv[4]);

    Mat vocabulary = loadYAML(vocabulary_path, DEFAULT_KEY);

    buildTrainingSet(dataset_images_path, vocabulary, dataset_csv_path, proportion);

    return 0;
}