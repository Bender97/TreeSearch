//
// Created by fusy on 14/06/20.
//

#include <iostream>
using namespace std;

#include "../../include/VocabularyUtility.h"

int main(int argc, char ** argv) {

    cv::String keys =
            "{ds                |../data/vocabulary_dataset/                  | vocabulary path}"
            "{voc               |../data/vocabulary/vocabulary.yml   | svm model path}"
            "{n                 |200                                    | bags number}"
            "{help h usage ?    |  | usage: vocabulary_images_path vocabulary_output_path bags_number\n"
            "if path contains a space, you should embrace path in quotes}";

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    string vocabulary_images_path = parser.get<cv::String>("ds");
    string vocabulary_output_path = parser.get<cv::String>("voc");
    int bags_number = parser.get<int>("n");

    /** CHECKING INPUT parameters **/
    if (vocabulary_images_path[vocabulary_images_path.length()-1] != '/') {
        cout << "Are you sure that this path is correct? " << vocabulary_images_path << endl;
        cout << "(maybe you should end it with / ) " << endl;
    }
    cout << "Starting building vocabulary from " << vocabulary_images_path
            << " with " << bags_number << " bags" << endl;

    storeVocabulary(vocabulary_images_path, vocabulary_output_path, bags_number);

    cout << "Vocabulary built and saved to " << vocabulary_output_path << endl << endl;

    return 0;
}