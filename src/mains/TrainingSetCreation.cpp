//
// Created by cogny on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/VocabularyUtility.h"
#include "../../include/DatasetUtility.h"

int main(int argc, char ** argv)
{
    cv::String keys =
            "{img_ds|../data/images_dataset/ | path to dataset of images}"
            "{voc   |../data/vocabulary/vocabulary_200.yml | vocabulary path}"
            "{csv_ds|../data/csv_dataset/ | path to csv dataset output folder }"
            "{p     |0.8       | train/test proportion}"
            "{help h usage ?    |      | usage: img_ds=dataset_images_path voc=vocabulary_path csv_ds=dataset_csv_path p=proportion\n"
            "if path contains a space, you should embrace path in quotes}";

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    string dataset_images_path = parser.get<cv::String>("img_ds");
    string vocabulary_path = parser.get<cv::String>("voc");
    string dataset_csv_path = parser.get<cv::String>("csv_ds");

    float proportion = parser.get<float>("p");

    cout << "Loading vocabulary: " << vocabulary_path << endl;
    Mat vocabulary = loadYAML(vocabulary_path, DEFAULT_KEY);
    cout << "Vocabulary correctly loaded" << endl << endl;

    buildTrainingSet(dataset_images_path, vocabulary, dataset_csv_path, proportion);

    cout << "Train and Test sets correctly built" << endl << endl;
    return 0;
}