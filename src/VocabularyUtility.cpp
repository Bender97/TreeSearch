//
// Created by fusy on 14/06/20.
//

#include "../include/VocabularyUtility.h"

bool storeVocabulary(string path_imgs, string voc, int n_bags) {
    return true;
}

bool storeVocabulary(string path_imgs, string voc) {
    return storeVocabulary(path_imgs, voc, DEFAULT_BAGS);
}

static Mat makeVocabulary(string path_imgs, int n_bags) {
    return Mat();
}
static Mat makeVocabulary(string path_imgs) {
    return makeVocabulary(path_imgs, DEFAULT_BAGS);
}