//
// Created by fusy on 13/06/20.
//

#include "../include/ImagesUtility.h"

vector<string> ImagesUtility::loadImagesPaths(string path) {

    vector<string> filenames;   // vector of all the filenames found in the specified folder
    vector<string> allowedExtensions = { ".jpg", ".jpeg", ".JPG", ".png", ".bmp" };    // allowed extensions for images

    /// for each extension, search for images in the specified folder
    for (int i = 0; i < allowedExtensions.size(); i++) {
        vector<string> currentFilenames;    // filenames found with the i-th extension

        /// look for matches with the pattern
        cv::glob(path + "*" + allowedExtensions[i],
                 currentFilenames, true);

        /// insert found filenames in the general vector
        filenames.insert(
                filenames.end(),
                currentFilenames.begin(),
                currentFilenames.end());
    }

    if (filenames.empty()) cout << "No images found. Exiting." << endl;

    return filenames;
}

Mat ImagesUtility::loadImage(string path) {
    return imread(path);
}

bool ImagesUtility::storeImage(string path, Mat img) {
    return imwrite(path, img);
}


