//
// Created by fusy on 14/06/20.
//

#include "../include/VocabularyUtility.h"

bool storeVocabulary(string path_imgs, string path_voc, int n_bags) {

    writeYAML(path_voc, DEFAULT_KEY, makeVocabulary(path_imgs, n_bags));
    return true;
}

bool storeVocabulary(string path_imgs, string path_voc) {
    return storeVocabulary(path_imgs, path_voc, DEFAULT_BAGS);
}

Mat makeVocabulary(string path_imgs, int n_bags) {
    //vector<Mat> sample_imgs = loadImage(path_imgs);
    vector<string> sample_imgs = loadImagesPaths(path_imgs);

    //to store the current input image
    Mat input;
    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;
    //To store the SIFT descriptor of current image
    Mat descriptor;
    //To store all the descriptors that are extracted from all the images.
    Mat featuresUnclustered;
    //The SIFT feature extractor and descriptor
    Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();

    //I select 20 (1000/50) images from 1000 images to extract
    //feature descriptors and build the vocabulary
    for (int i=0; i<sample_imgs.size(); i++) {
        Mat input = imread(sample_imgs[i]);

        cvtColor(input, input, COLOR_BGR2GRAY);
        //detect feature points
        detector->detect(input, keypoints);
        //compute the descriptors for each keypoint
        detector->compute(input, keypoints, descriptor);
        //put the all feature descriptors in a single Mat object
        featuresUnclustered.push_back(descriptor);
        //print the progressm
        cout << "image " << i << "/" << sample_imgs.size() << " parsed" << endl;
    }

    //Construct BOWKMeansTrainer
    //the number of bags
    int dictionarySize = n_bags;
    //define Term Criteria
    TermCriteria tc(TermCriteria::Type::MAX_ITER, 100, 0.001);
    //retries number
    int retries = 1;
    //necessary flags
    int flags = KMEANS_PP_CENTERS;
    //Create the BoW (or BoF) trainer
    BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
    //cluster the feature vectors
    Mat dictionary = bowTrainer.cluster(featuresUnclustered);

    return dictionary;
}
Mat makeVocabulary(string path_imgs) {
    return makeVocabulary(path_imgs, DEFAULT_BAGS);
}